"""
========== 全局异常处理器 ==========
将所有自定义异常（及未预期的系统异常）转换为统一的 ErrorResponse JSON 格式。

设计原则：
1. 自定义异常（RAGException 子类）：使用异常实例中预设的 error_code / message / status_code
2. HTTP 异常（HTTPException）：透传 FastAPI 原生的错误响应
3. 未预期异常（Exception）：兜底捕获，返回通用的"服务器内部错误"，不泄露堆栈信息
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.exceptions import (
    RAGException,
    RetrievalException,
    GenerationException,
    DocumentProcessingException,
    VectorStoreException,
    ConfigurationException,
)
from app.schemas import ErrorResponse

# 使用 Python 标准 logging，输出到控制台（uvicorn 会捕获并显示）
logger = logging.getLogger("rag")


def register_exception_handlers(app: FastAPI) -> None:
    """
    向 FastAPI 应用注册所有异常处理器。
    调用一次即可，应在 app 启动时调用。

    异常处理优先级（FastAPI 按注册的反向顺序匹配）：
    1. RAGException（及其子类）→ JSON 错误响应
    2. RequestValidationError → 422 参数校验失败
    3. Exception → 500 兜底
    """

    # ----- 自定义异常的统一入口 -----
    @app.exception_handler(RAGException)
    async def rag_exception_handler(request: Request, exc: RAGException):
        """
        捕获所有 RAGException 及其子类。
        因为 RetrievalException、GenerationException 等都是 RAGException 的子类，
        所以它们全部会被这个 handler 统一处理。
        """
        # 使用 exc 中预设的 status_code，方便在日志中区分不同严重级别
        if exc.status_code >= 500:
            logger.error(
                "[%s] %s | detail=%s | path=%s",
                exc.error_code, exc.message, exc.detail, request.url.path,
            )
        else:
            # 4xx 类错误（如不支持的文件格式）属于客户端问题，用 warning 级别
            logger.warning(
                "[%s] %s | detail=%s | path=%s",
                exc.error_code, exc.message, exc.detail, request.url.path,
            )

        # FastAPI 的 JSONResponse 自动将 Pydantic 模型序列化为 JSON
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
                detail=exc.detail or "",
            ).model_dump(),
        )

    # ----- 请求参数校验失败 -----
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        当请求参数不符合 Pydantic schema 时，FastAPI 默认会返回 422 和
        冗长的错误详情。这里将错误信息简化为更友好的格式。
        """
        # 提取每个字段的校验错误信息
        errors = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            errors.append(f"{field}: {msg}")

        logger.warning("参数校验失败 | path=%s | errors=%s", request.url.path, errors)

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message="请求参数校验失败",
                detail="; ".join(errors),
            ).model_dump(),
        )

    # ----- 最终兜底：未预期的系统异常 -----
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        捕获所有未被上面 handler 覆盖的异常。
        这是最后的安全网，防止堆栈信息泄露到前端。
        """
        # 使用 exc_info=True 记录完整堆栈，方便排查
        logger.exception(
            "未预期的服务器内部错误 | path=%s | type=%s",
            request.url.path, type(exc).__name__,
        )

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message="服务器内部错误，请稍后重试",
                detail="",
            ).model_dump(),
        )
