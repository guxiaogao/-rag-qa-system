"""
========== 数据模型模块 ==========
定义 API 的请求和响应数据结构。
使用 Pydantic 进行数据验证。
"""

from typing import List
from pydantic import BaseModel

from app.config import settings

# ---------- 文档来源信息 ----------
class Source(BaseModel):
    """单个检索来源的信息"""
    content: str       # 检索到的文本片段
    filename: str      # 来源文件名
    chunk_index: int   # 该片段在文档中的序号
    score: float       # 相关性得分
    source_type: str = "knowledge_base"   # 来源类型: "knowledge_base" | "web"
    source_url: str = ""                  # Web 来源的原始 URL（KB 来源为空）


# ---------- 问答接口 ----------

class ChatMessage(BaseModel):
    """单条对话消息"""
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """问答请求

    服务端已默认开启流式输出 + 重排序 + 查询重写，前端无需再传这些开关。
    extra="ignore" 允许调用方无意中传入旧字段时不报错，平滑兼容。
    """
    model_config = {"extra": "ignore"}
    query: str                          # 用户问题
    top_k: int = settings.top_k         # 检索返回的文档片段数（默认从全局配置读取）
    use_mmr: bool = False               # 是否使用 MMR 增加多样性
    temperature: float = settings.llm_temperature  # LLM 生成温度（0-1）
    conversation_history: list[ChatMessage] = []  # 多轮对话历史（按时间顺序，最近 3 轮即可）
    force_web_search: bool = False          # 前端勾选"联网搜索"后强制搜网，忽略阈值判断


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str                        # 生成的回答
    sources: List[Source]              # 检索来源（用于展示和验证）
    llm_model: str                     # 使用的模型名称
    web_search_used: bool = False      # 本次回答是否触发了 Web 搜索 fallback


# ---------- 仅检索接口（调试用） ----------
class SearchRequest(BaseModel):
    """检索请求"""
    model_config = {"extra": "ignore"}
    query: str                          # 搜索关键词
    top_k: int = settings.top_k         # 返回的片段数（默认从全局配置读取）
    use_mmr: bool = False               # 是否使用 MMR 增加多样性


class SearchResponse(BaseModel):
    """检索响应"""
    results: List[Source]     # 检索结果


# ---------- 文档管理接口 ----------
class DocumentInfo(BaseModel):
    """文档信息"""
    id: str                   # 文档在 ChromaDB 中的 ID
    filename: str             # 原始文件名
    chunk_count: int          # 该文档被切分成的片段数


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentInfo]
    total: int                # 文档总数


class DeleteResponse(BaseModel):
    """删除响应"""
    message: str
    document_id: str


# ---------- 错误响应 ----------
class ErrorResponse(BaseModel):
    """
    统一的 API 错误响应格式。
    所有异常经过全局异常处理器后，都以此格式返回给前端。
    """
    error_code: str           # 错误码（英文标识，如 RETRIEVAL_ERROR，方便前端判断分支）
    message: str              # 用户友好的错误描述
    detail: str = ""          # 可选的详细信息（如具体的异常原因）


# ---------- 系统 ----------
class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    vector_store_size: int    # 向量数据库中的文档片段总数
