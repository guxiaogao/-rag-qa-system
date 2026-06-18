"""
========== API 路由模块 ==========
定义所有 RESTful API 端点。
使用 FastAPI Router 组织路由。

异常处理策略：
- 底层模块（retriever, generator, database, document_loader）已将各自异常
  包装为自定义异常（RetrievalException, GenerationException 等）。
- 路由层不重复捕获这些自定义异常，它们会自然传播到全局异常处理器，
  由 error_handlers.py 统一返回友好 JSON 响应。
- 路由层只处理 HTTP 级别的异常（如 HTTPException 参数校验）。
- 流式端点（/chat?stream=true）在 SSE 生成器内部自行捕获错误，
  以 SSE 格式返回，而非走全局 JSON 异常处理器。
"""

import os
import uuid
import logging
import json
import asyncio

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List

from app.schemas import (
    ChatRequest, ChatResponse, Source,
    SearchRequest, SearchResponse,
    DocumentInfo, DocumentListResponse,
    UploadResponse, DeleteResponse, HealthResponse,
)
from app.retriever import retrieve, get_all_documents, delete_document
from app.generator import generate_answer, generate_answer_stream
from app.document_loader import load_and_split
from app.database import get_vector_store
from app.config import settings, PROJECT_ROOT, limiter
from app.exceptions import GenerationException

# 路由层日志
logger = logging.getLogger("rag.router")

# 创建路由器，所有 API 路径以 /api 开头
router = APIRouter(prefix="/api", tags=["RAG API"])


# ========== Web 搜索 fallback 决策门 ==========

def _resolve_docs(query: str, docs: list, top_k: int) -> tuple:
    """
    检查知识库检索质量，必要时自动 fallback 到 Web 搜索补全上下文。

    router 非流式、流式两条路径共享同一决策逻辑，避免重复代码。

    决策矩阵：
        1. web_search_enabled=False → 原样返回 KB docs
        2. KB 空       → 直接走 Web 搜索
        3. KB 最高分 < 阈值 → KB + Web 合并
        4. KB 最高分 ≥ 阈值 → 原样返回 KB docs

    返回: (docs, web_used)
    """
    if not settings.web_search_enabled:
        return docs, False

    # 空知识库：直接走 Web
    if not docs:
        from app.web_search import web_search
        web_docs = web_search(query, num_results=top_k)
        return web_docs, bool(web_docs)

    # 有结果，检查最高分
    max_score = max(
        d.metadata.get("rerank_score", d.metadata.get("score", 0.0))
        for d in docs
    )
    if max_score >= settings.web_search_fallback_threshold:
        return docs, False

    # KB 质量不足，Web 补全
    from app.web_search import web_search
    web_docs = web_search(query, num_results=max(3, top_k - len(docs)))
    if not web_docs:
        logger.info("Web 搜索无结果，继续使用知识库结果")
        return docs, False

    logger.info(
        "Web fallback 触发：KB max_score=%.3f < threshold=%.3f，合并 %d 条 web 结果",
        max_score, settings.web_search_fallback_threshold, len(web_docs),
    )
    return docs + web_docs, True


# ========== 健康检查 ==========
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    检查服务是否正常运行，返回向量库中的文档数。
    如果向量库暂时不可用，不抛出异常，而是返回 count=0（服务降级）。
    """
    try:
        vs = get_vector_store()
        count = vs._collection.count()
    except Exception:
        # 向量库不可用时静默降级，不阻止健康检查返回
        # 但记录警告日志，方便运维排查
        logger.warning("健康检查：向量库暂时不可用，返回 count=0", exc_info=True)
        count = 0
    return HealthResponse(status="ok", vector_store_size=count)


# ========== 问答接口 ==========
@router.post("/chat")
@limiter.limit("60/minute")
async def chat(body: ChatRequest, request: Request):
    """
    核心问答接口。

    支持两种模式：
    - stream=false（默认）：同步返回完整 JSON 响应（向后兼容）
    - stream=true：返回 SSE 事件流，逐 token 推送生成结果

    流程：
    1. 根据问题检索相关文档片段
    2. 将问题 + 片段交给 LLM 生成回答
    3. 返回回答和检索来源

    异常处理：
    - 非流式：检索/生成失败 → 全局异常处理器 → JSON 错误响应
    - 流式（检索阶段）：检索失败 → 全局异常处理器 → JSON 错误（SSE 尚未开始）
    - 流式（生成阶段）：生成失败 → SSE 错误事件 + done（SSE 已开始，必须用 SSE 格式）
    """
    # ========== 非流式模式 ==========
    if not body.stream:
        # 将 Pydantic 模型转为 dict，供 generator 使用
        history_dicts = [m.model_dump() for m in body.conversation_history]

        docs = await asyncio.to_thread(
            retrieve,
            query=body.query,
            top_k=body.top_k,
            use_mmr=body.use_mmr,
            use_reranker=body.use_reranker,
            use_rewrite=body.use_rewrite,
        )

        # ---- Web 搜索 fallback ----
        docs, web_used = _resolve_docs(
            query=body.query,
            docs=docs,
            top_k=body.top_k,
        )

        # 调用 LLM 生成回答
        answer = await asyncio.to_thread(
            generate_answer,
            query=body.query,
            docs=docs,
            temperature=body.temperature,
            conversation_history=history_dicts,
        )

        sources = [
            Source(
                content=doc.page_content,
                filename=doc.metadata.get("filename", "未知"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                score=doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)),
                source_type=doc.metadata.get("source_type", "knowledge_base"),
                source_url=doc.metadata.get("source_url", ""),
            )
            for doc in docs
        ]
        return ChatResponse(
            answer=answer,
            sources=sources,
            llm_model=settings.chat_model,
            web_search_used=web_used,
        )

    # ========== 流式模式 ==========
    # 将 Pydantic 模型转为 dict，供 generator 使用
    history_dicts = [m.model_dump() for m in body.conversation_history]

    # 检索在 SSE 开始前执行（同步函数送入线程池，避免阻塞事件循环）
    docs = await asyncio.to_thread(
        retrieve,
        query=body.query,
        top_k=body.top_k,
        use_mmr=body.use_mmr,
        use_reranker=body.use_reranker,
        use_rewrite=body.use_rewrite,
    )

    # ---- Web 搜索 fallback ----
    docs, web_used = _resolve_docs(
        query=body.query,
        docs=docs,
        top_k=body.top_k,
    )

    # 预先格式化来源数据
    final_docs = docs
    sources_data = [
        Source(
            content=doc.page_content,
            filename=doc.metadata.get("filename", "未知"),
            chunk_index=doc.metadata.get("chunk_index", 0),
            score=doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)),
            source_type=doc.metadata.get("source_type", "knowledge_base"),
            source_url=doc.metadata.get("source_url", ""),
        ).model_dump()
        for doc in final_docs
    ]

    async def event_generator(
        _docs=final_docs,
        _sources=sources_data,
        _web_used=web_used,
        _query=body.query,
        _temperature=body.temperature,
        _history=history_dicts,
    ):
        """SSE 事件生成器：token → sources → web_search_used → done

        使用默认参数显式绑定闭包变量（Python 在函数定义时求值默认参数），
        防御未来重构时将本函数提取到 router 外导致延迟绑定的 bug。
        """
        try:
            # 逐 token 推送生成结果
            async for token in generate_answer_stream(
                query=_query,
                docs=_docs,
                temperature=_temperature,
                conversation_history=_history,
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            # 推送检索来源
            yield f"data: {json.dumps({'type': 'sources', 'sources': _sources}, ensure_ascii=False)}\n\n"

            # 推送 Web 搜索标记
            yield f"data: {json.dumps({'type': 'meta', 'web_search_used': _web_used}, ensure_ascii=False)}\n\n"

            # 结束标记
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except GenerationException as e:
            yield f"data: {json.dumps({'type': 'error', 'error_code': e.error_code, 'message': e.message, 'detail': e.detail or ''}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.exception("流式生成过程中发生未预期的错误")
            yield f"data: {json.dumps({'type': 'error', 'error_code': 'INTERNAL_ERROR', 'message': '服务器内部错误，请稍后重试'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        content=event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ========== 仅检索接口（调试用）==========
@router.post("/search", response_model=SearchResponse)
@limiter.limit("120/minute")
async def search(body: SearchRequest, request: Request):
    """
    仅检索不生成回答，用于调试和查看检索效果。

    异常处理：检索失败时，RetrievalException 传播到全局处理器。
    """
    docs = await asyncio.to_thread(
        retrieve,
        query=body.query,
        top_k=body.top_k,
        use_mmr=body.use_mmr,
        use_reranker=body.use_reranker,
        use_rewrite=body.use_rewrite,
    )
    results = [
        Source(
            content=doc.page_content,
            filename=doc.metadata.get("filename", "未知"),
            chunk_index=doc.metadata.get("chunk_index", 0),
            score=doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)),
        )
        for doc in docs
    ]
    return SearchResponse(results=results)


# ========== 文档管理 ==========
@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    列出向量库中所有已索引的文档。

    异常处理：向量库读取失败时，RetrievalException 传播到全局处理器。
    """
    docs = get_all_documents()
    doc_list = [DocumentInfo(**d) for d in docs]
    return DocumentListResponse(documents=doc_list, total=len(doc_list))


@router.post("/documents/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    上传文档并自动索引到向量库。

    支持格式：.txt, .md, .pdf（文件大小上限见配置 MAX_UPLOAD_SIZE_MB）。
    流程：验证类型 → 验证大小 → 保存临时文件 → 加载 → 分块 → 存入 ChromaDB

    异常：
    - HTTPException(400)：文件格式不支持或文件过大
    - DocumentProcessingException：文档加载或分块失败
    - VectorStoreException：向量库存入失败
    以上异常均由全局异常处理器统一格式化返回。
    """
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    # 验证文件类型（在 router 层做，属于 API 参数校验，用 HTTPException 最合适）
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".txt", ".md", ".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式：{ext}，仅支持 .txt, .md, .pdf",
        )

    # 验证文件大小（读取内容后检查，防止超大文件耗尽 embedding 配额）
    content = await file.read()
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大（{len(content) / 1024 / 1024:.1f}MB），"
                   f"上限为 {settings.max_upload_size_mb}MB",
        )

    # 保存临时文件
    temp_dir = os.path.join(PROJECT_ROOT, "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        # 加载、分块、存入向量库
        # 如果此处抛出 DocumentProcessingException 或 VectorStoreException，
        # 会被全局异常处理器捕获并返回友好响应
        chunks = load_and_split(temp_path)
        doc_id = str(uuid.uuid4())
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id

        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        return UploadResponse(
            message="文档上传并索引成功",
            filename=file.filename or "unknown",
            chunk_count=len(chunks),
            document_id=doc_id,
        )
    finally:
        # 无论成功还是失败，都要清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_document_by_filename(filename: str):
    """
    删除指定文件的所有向量索引。

    异常：
    - HTTPException(404)：文件不存在
    - RetrievalException：向量库操作失败
    """
    deleted = delete_document(filename)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"未找到文件：{filename}",
        )
    return DeleteResponse(
        message=f"已删除 {deleted} 个文档片段",
        document_id=filename,
    )
