"""
========== 配置管理模块 ==========
从环境变量加载配置，提供统一的配置入口。
使用 pydantic-settings 自动加载 .env 文件。
"""

import os
from pydantic_settings import BaseSettings

# 项目根目录的绝对路径（基于 config.py 位置推算，不受 cwd 影响）
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# .env 文件绝对路径
_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")


class Settings(BaseSettings):
    # ========== DashScope 通义千问 API 配置 ==========
    # API Key，在 https://dashscope.aliyun.com/ 申请
    dashscope_api_key: str = ""

    # ========== 模型配置 ==========
    # 生成答案用的模型（qwen-max 旗舰模型，表达自然细腻，推荐用于对回答风格有要求的场景）
    chat_model: str = "qwen-max"
   
    # 向量化用的 embedding 模型
    embedding_model: str = "text-embedding-v3"
  
    # 评估时做裁判用的模型（用便宜的就行）
    judge_model: str = "qwen-turbo"

    # ========== 检索参数 ==========
    # 文档分块大小（字符数）
    chunk_size: int = 500

    # 分块重叠（避免切断关键信息）
    chunk_overlap: int = 100

    # 检索返回的候选 chunk 数量
    top_k: int = 5

    # ========== 重排序参数（DashScope Rerank API）==========
    # 是否启用重排序（服务端总开关，关闭时即使 API 传 use_reranker=true 也不生效）
    rerank_enabled: bool = True

    # 重排序模型名称（DashScope 模型 ID）
    # gte-rerank 是阿里通义实验室的中英双语重排序模型，效果与 bge-reranker-v2-m3 同级别
    rerank_model: str = "gte-rerank"

    # 重排序时第一阶段检索的候选数量（先取这么多，再用 Rerank API 精排取 top_k）
    rerank_fetch_k: int = 20

    # ========== Query Rewrite 参数 ==========
    # 是否启用查询重写（服务端总开关，关闭时即使 API 传 use_rewrite=true 也不生效）
    rewrite_enabled: bool = True

    # 用于查询重写的模型（用便宜的就行，qwen-turbo 足够做关键词提取）
    rewrite_model: str = "qwen-turbo"

    # ========== LLM 生成参数 ==========
    # 生成温度（0-1），越高越随机、越低越确定
    llm_temperature: float = 0.8

    # ========== Embedding 参数 ==========
    # Embedding API 单次请求最大文本数（百炼平台限制 10）
    embedding_batch_size: int = 10

    # ========== Web 搜索配置 ==========
    # 是否启用 Web 搜索 fallback（服务端总开关）。
    # 开启后，KB 检索结果质量不足时自动搜索互联网补全上下文。
    web_search_enabled: bool = False

    # 知识库检索相关性阈值（0-1）
    # KB 最高分低于此值时，自动触发 Web 搜索补全上下文；
    # 设为 0 表示永远触发；设为 1 表示永不触发
    web_search_fallback_threshold: float = 0.3

    # Web 搜索结果数量
    web_search_num_results: int = 5

    # ========== CORS 配置 ==========
    # 允许的跨域来源，逗号分隔多个域名。
    # 开发/本地调试使用 "*"（默认）；生产环境请设置为具体域名，如 "https://your-app.example.com"
    cors_origins: str = "*"

    # ========== 系统配置 ==========
    # 向量数据库持久化路径
    chroma_db_path: str = os.path.join(PROJECT_ROOT, "data", "chroma_db")
   
    # ChromaDB 集合名称
    chroma_collection_name: str = "rag_documents"
    
    # DashScope OpenAI 兼容接口地址（不需要改）
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ========== 上传限制 ==========
    # 单次上传文件大小上限（MB）
    max_upload_size_mb: int = 10

    class Config:
        env_file = _ENV_FILE
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略 .env 中未定义的字段，避免新增 key 导致启动失败


# 全局单例配置
settings = Settings()

# ========== 限流器 ==========
# 单例 Limiter，供 main.py 注册中间件、router.py 装饰端点使用。
# key_func=get_remote_address 按客户端 IP 独立计数。
from slowapi import Limiter as _Limiter
from slowapi.util import get_remote_address as _get_remote_address

limiter = _Limiter(key_func=_get_remote_address, default_limits=["200/minute"])
