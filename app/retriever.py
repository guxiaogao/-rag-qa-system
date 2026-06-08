"""
========== 检索模块 ==========
负责从 ChromaDB 中找出与问题最相关的文档片段。
支持普通相似度检索、MMR（最大边际相关性）检索、以及查询重写。

管道：query rewrite（可选）→ vector search（MMR/相似度）→ rerank（可选）

异常处理策略：
- 所有底层异常（ChromaDB 连接失败、Embedding API 异常等）在此层
  统一包装为 RetrievalException，方便上层（router）集中处理。
- 如果向量库为空（无文档），视为正常情况，返回空列表而非抛异常。
- 查询重写失败时优雅降级，使用原始查询继续检索。
"""

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate

from app.database import get_vector_store
from app.config import settings
from app.exceptions import RetrievalException, VectorStoreException

logger = logging.getLogger("rag.retriever")

# ========== Query Rewrite Prompt ==========

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个查询优化助手。你的任务是将用户输入的自然语言问题，转换为更适合向量检索的关键词查询。

要求：
1. 提取问题中的核心概念和关键词
2. 将口语化表达转换为更正式、更精确的技术术语
3. 保留原始问题的语义，但去掉冗余的修饰词和语气词
4. 只输出重写后的查询文本，不要任何解释或额外内容"""),
    ("human", "原始查询：{query}\n\n重写后的查询："),
])


def rewrite_query(query: str) -> str:
    """
    使用廉价 LLM 将用户查询重写为检索友好的关键词查询。

    参数：
        query: 用户原始问题

    返回：
        重写后的查询字符串。
        如果重写失败或返回空结果，返回原始查询（优雅降级，不抛异常）。
    """
    if not query or not query.strip():
        return query

    try:
        llm = ChatOpenAI(
            model=settings.rewrite_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            temperature=0.0,  # 确定性输出
        )
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        rewritten = response.content.strip()

        if not rewritten:
            logger.warning("Query rewrite 返回空结果，降级使用原始查询")
            return query

        logger.info("Query rewrite: '%s' -> '%s'", query[:80], rewritten[:80])
        return rewritten

    except Exception as e:
        logger.warning(
            "Query rewrite 失败 (%s: %s)，降级使用原始查询",
            type(e).__name__, str(e)[:100],
        )
        return query


def retrieve(
    query: str,
    top_k: int,  # 必传，不再设默认值——每个调用方应显式决定检索数量
    use_mmr: bool = False,
    fetch_k: Optional[int] = None,
    use_reranker: bool = False,
    use_rewrite: bool = False,
) -> List[LCDocument]:
    """
    检索与问题最相关的文档片段。

    参数：
        query:        用户问题
        top_k:        返回多少个片段
        use_mmr:      是否使用 MMR（最大边际相关性）
        fetch_k:      MMR 时先取多少个候选（默认 top_k * 5）
        use_reranker: 是否使用 DashScope Rerank API 进行重排序。
        use_rewrite:  是否启用查询重写。
                      开启后，先用廉价 LLM 将用户查询转换为检索友好的关键词，
                      再用重写后的查询做向量检索。
                      需要 settings.rewrite_enabled 为 True 才实际生效。
                      失败时自动降级使用原始查询。

    返回：
        LangChain Document 列表，每个包含 page_content 和 metadata。

    异常：
        RetrievalException：检索过程发生错误（数据库连接、Embedding 失败等）
    """
    # 查询重写需要同时满足：用户请求 + 服务端开关
    should_rewrite = use_rewrite and settings.rewrite_enabled
    # 重排序需要同时满足：用户请求 + 服务端开关
    should_rerank = use_reranker and settings.rerank_enabled
    # 重排序时第一阶段取更多候选，否则直接取 top_k
    first_stage_k = settings.rerank_fetch_k if should_rerank else top_k

    try:
        vector_store = get_vector_store()
    except VectorStoreException as e:
        raise RetrievalException(
            detail=f"获取向量库连接失败: {e.detail or e.message}",
        )

    try:
        # ---- 查询重写阶段 ----
        if should_rewrite:
            query = rewrite_query(query)

        if use_mmr:
            # MMR 检索：平衡相关性和多样性
            # 注意：max_marginal_relevance_search 返回的 Document
            # 不携带 relevance score。需要先做一次相似度检索获取分数，
            # 再将分数映射到 MMR 结果上。
            if fetch_k is None:
                fetch_k = first_stage_k * 5

            # 先获取候选及分数
            scored = vector_store.similarity_search_with_relevance_scores(
                query=query, k=fetch_k,
            )
            content_to_score: dict[str, float] = {
                doc.page_content: round(score, 4)
                for doc, score in scored
            }

            # 再用 MMR 重排（不需要再查询——同一 query 下候选集一致）
            mmr_docs = vector_store.max_marginal_relevance_search(
                query=query, k=first_stage_k, fetch_k=fetch_k,
            )

            # 将相似度分数映射回 MMR 结果
            docs = []
            for doc in mmr_docs:
                doc.metadata["score"] = content_to_score.get(doc.page_content, 0.0)
                docs.append(doc)
        else:
            # 普通相似度检索
            docs = vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=first_stage_k,
            )
            result = []
            for doc, score in docs:
                doc.metadata["score"] = round(score, 4)
                result.append(doc)
            docs = result

        # ---- 重排序阶段 ----
        if should_rerank:
            from app.reranker import rerank
            try:
                docs = rerank(query=query, docs=docs, top_k=top_k)
            except RuntimeError as e:
                logger.warning(
                    "Rerank 执行失败 (%s)，降级使用原始排序",
                    str(e)[:120],
                )
                # 重排序失败时，回退到原始相似度排序的前 top_k 个
                docs = sorted(docs, key=lambda d: d.metadata.get("score", 0.0), reverse=True)[:top_k]

        return docs

    except RetrievalException:
        raise
    except Exception as e:
        raise RetrievalException(
            detail=f"检索执行失败 ({type(e).__name__}): {str(e)}",
        )


def get_all_documents() -> List[dict]:
    """
    获取向量库中所有文档的元数据摘要。
    用于文档管理接口。
    """
    try:
        collection = get_vector_store()._collection
        all_data = collection.get()
    except Exception as e:
        raise RetrievalException(
            detail=f"获取文档列表失败 ({type(e).__name__}): {str(e)}",
        )

    files = {}
    for i, doc_id in enumerate(all_data["ids"]):
        meta = all_data["metadatas"][i]
        filename = meta.get("filename", "unknown")
        if filename not in files:
            files[filename] = {"id": doc_id, "chunks": 0}
        files[filename]["chunks"] += 1
    return [
        {"id": v["id"].split("_")[0] if "_" in v["id"] else v["id"],
         "filename": k, "chunk_count": v["chunks"]}
        for k, v in files.items()
    ]


def delete_document(filename: str) -> int:
    """
    在向量库中删除指定文件的所有片段。
    返回被删除的片段数。
    """
    try:
        collection = get_vector_store()._collection
        all_data = collection.get()
        ids_to_delete = []
        for i, doc_id in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            if meta.get("filename") == filename:
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except RetrievalException:
        raise
    except Exception as e:
        raise RetrievalException(
            detail=f"删除文档失败 ({type(e).__name__}): {str(e)}",
        )
