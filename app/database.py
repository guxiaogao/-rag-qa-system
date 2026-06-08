"""
========== 向量数据库模块 ==========
管理 ChromaDB 的连接和操作。
使用 LangChain 的 Chroma 封装，支持持久化存储。
注意：不能直接用 OpenAIEmbeddings 对接 DashScope，因为
langchain-openai 内部会 tokenize 后发送 token ID 数组，
而 DashScope API 只接受原始文本字符串。所以自定义 Embedding 类。
"""

from typing import List
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.config import settings
from app.exceptions import VectorStoreException, ConfigurationException


class DashScopeEmbeddings(Embeddings):
    """
    自定义 Embedding 类，直接通过 OpenAI 兼容接口调用 DashScope。
    关键区别：发送原始文本字符串而非 tokenized 数组。
    """

    def __init__(self, model: str = None):
        self.model = model or settings.embedding_model
        # 如果 API Key 为空，后续调用 embedding API 时会失败。
        # 这里在初始化时不校验，由上层 (retriever / generator) 调用时捕获异常并转换。
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文本批量向量化，自动分批以适配 API 的 batch size 限制（text-embedding-v2/v3 最高 10）。
        如果 API 调用失败（Key 无效、超时、限流等），异常会被上层捕获并转为 VectorStoreException。
        """
        all_embeddings = []
        for i in range(0, len(texts), settings.embedding_batch_size):
            batch = texts[i : i + settings.embedding_batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本向量化。
        如果 API 调用失败，异常会被上层捕获并转为 RetrievalException。
        """
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


def get_embedding_model() -> DashScopeEmbeddings:
    """获取 DashScope Embedding 模型实例"""
    if not settings.dashscope_api_key:
        raise ConfigurationException(
            message="API Key 未配置，请在 .env 文件中设置 DASHSCOPE_API_KEY",
            detail="DashScope API Key 为空，无法初始化 Embedding 模型",
        )
    return DashScopeEmbeddings()


# 模块级缓存：避免每次调用都创建新的 Chroma/OpenAI 实例
_vector_store: Chroma | None = None


def get_vector_store() -> Chroma:
    """
    获取（或创建）持久化的 ChromaDB 向量存储实例。
    如果数据库已存在则直接加载，否则自动创建新库。
    使用模块级缓存避免重复创建连接，在高并发场景下显著减少开销。

    可能的异常：
    - ConfigurationException：API Key 未配置（由 get_embedding_model 抛出）
    - VectorStoreException：ChromaDB 初始化或连接失败
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    try:
        embedding = get_embedding_model()
        _vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embedding,
            persist_directory=settings.chroma_db_path,
        )
        return _vector_store
    except (ConfigurationException, VectorStoreException):
        # 这两类已经是自定义异常，直接向上传播
        raise
    except Exception as e:
        # 将 ChromaDB 或其他底层库的原始异常包装为 VectorStoreException
        raise VectorStoreException(
            detail=f"ChromaDB 初始化失败: {type(e).__name__}: {str(e)}",
        )


def reset_collection() -> None:
    """
    清空（删除并重建）向量数据库集合。
    用于 init_db.py 重新索引时清理旧数据。
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        try:
            client.delete_collection(settings.chroma_collection_name)
        except Exception:
            # 集合不存在时忽略
            pass
        # 清除模块缓存，下次 get_vector_store() 会创建新集合
        global _vector_store
        _vector_store = None
    except Exception as e:
        raise VectorStoreException(
            detail=f"重置向量库失败: {type(e).__name__}: {str(e)}",
        )


def get_collection():
    """
    直接获取 ChromaDB 原生 Collection 对象。
    用于需要直接操作元数据或 ID 的场景。
    """
    try:
        vs = get_vector_store()
        return vs._collection
    except (ConfigurationException, VectorStoreException):
        raise
    except Exception as e:
        raise VectorStoreException(
            detail=f"获取 ChromaDB Collection 失败: {type(e).__name__}: {str(e)}",
        )


def get_indexed_filenames() -> set[str]:
    """
    查询向量库中已索引的文件名集合。
    用于增量索引：跳过已入库的文件，只处理新文件。

    返回：
        set[str]: 已索引的文件名集合（空集合表示首次运行或库为空）

    注意：
        仅比较文件名，不校验文件内容是否变化。
        如需更新已索引文件的内容，请用 --full 重建。
    """
    try:
        collection = get_vector_store()._collection
        all_data = collection.get()
        filenames = set()
        for meta in all_data["metadatas"]:
            name = meta.get("filename", "")
            if name:
                filenames.add(name)
        return filenames
    except (ConfigurationException, VectorStoreException):
        raise
    except Exception as e:
        raise VectorStoreException(
            detail=f"查询已索引文件列表失败: {type(e).__name__}: {str(e)}",
        )
