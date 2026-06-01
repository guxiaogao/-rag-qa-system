"""
========== 自定义异常模块 ==========
定义 RAG 系统的异常层级。
每一层（检索、生成、文档处理、向量库）都有对应的异常类型，
方便在全局异常处理器中做差异化处理和返回合适的状态码。
"""


class RAGException(Exception):
    """
    RAG 系统的根基异常类。
    所有自定义异常都继承自它，便于全局异常处理器统一捕获。

    属性：
        error_code:  错误码（英文标识，便于前端判断）
        message:     用户友好的错误描述
        status_code: HTTP 状态码
        detail:      可选的详细错误信息（仅返回给前端，不记录日志）
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 500,
        detail: str = None,
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


# ========== 检索层异常 ==========

class RetrievalException(RAGException):
    """
    检索阶段发生的异常。
    典型场景：
    - 向量数据库连接失败
    - 查询向量化失败（Embedding API 异常）
    - 相似度搜索执行异常
    """

    def __init__(self, message: str = None, detail: str = None):
        super().__init__(
            error_code="RETRIEVAL_ERROR",
            message=message or "文档检索失败，请稍后重试",
            status_code=503,  # 503 Service Unavailable：依赖服务不可用
            detail=detail,
        )


# ========== 生成层异常 ==========

class GenerationException(RAGException):
    """
    LLM 生成答案阶段发生的异常。
    典型场景：
    - DashScope API Key 无效或过期
    - API 调用超时
    - API 限流（Rate Limit）
    - 模型不存在或不可用
    """

    def __init__(self, message: str = None, detail: str = None):
        super().__init__(
            error_code="GENERATION_ERROR",
            message=message or "LLM 生成回答失败，请检查 API 配置或稍后重试",
            status_code=503,
            detail=detail,
        )


# ========== 文档处理异常 ==========

class DocumentProcessingException(RAGException):
    """
    文档加载 / 解析 / 分块阶段发生的异常。
    典型场景：
    - 不支持的文件格式
    - PDF 解析失败（pypdf 未安装或文件损坏）
    - 文件读取 I/O 错误
    """

    def __init__(self, message: str = None, detail: str = None, status_code: int = 400):
        super().__init__(
            error_code="DOCUMENT_ERROR",
            message=message or "文档处理失败",
            status_code=status_code,
            detail=detail,
        )


# ========== 向量库异常 ==========

class VectorStoreException(RAGException):
    """
    向量数据库操作异常。
    典型场景：
    - ChromaDB 数据文件损坏
    - 集合操作失败（写入 / 删除）
    - 数据库路径不可访问
    """

    def __init__(self, message: str = None, detail: str = None):
        super().__init__(
            error_code="VECTOR_STORE_ERROR",
            message=message or "向量数据库操作失败，请稍后重试",
            status_code=503,
            detail=detail,
        )


# ========== 配置异常 ==========

class ConfigurationException(RAGException):
    """
    系统配置错误。
    典型场景：
    - 缺少 API Key
    - 模型名称配置错误
    - .env 文件缺失或格式错误
    """

    def __init__(self, message: str = None, detail: str = None):
        super().__init__(
            error_code="CONFIGURATION_ERROR",
            message=message or "系统配置错误，请联系管理员",
            status_code=500,
            detail=detail,
        )
