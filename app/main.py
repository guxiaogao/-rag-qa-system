"""
========== FastAPI 应用入口 ==========
创建 FastAPI 应用，注册路由，配置 CORS。
通过 `uvicorn app.main:app --reload` 启动。
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from slowapi.middleware import SlowAPIMiddleware

from app.router import router
from app.error_handlers import register_exception_handlers
from app.config import PROJECT_ROOT, settings, limiter

# 创建 FastAPI 应用实例
app = FastAPI(
    title="RAG 智能问答系统",
    description="基于 FastAPI + LangChain + ChromaDB 的检索增强生成系统",
    version="1.0.0",
)

# 配置 CORS
# cors_origins 支持两种形式：
#   "*"               → 开发/本地调试，允许所有来源（注意：此时 allow_credentials 必须为 False）
#   "https://a.com,https://b.com"  → 生产环境，逗号分隔的具体域名列表
_cors_raw = settings.cors_origins.strip()
if _cors_raw == "*":
    _origins = ["*"]
    _allow_credentials = False
else:
    _origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册限流中间件（必须在 CORS 之后，否则预检请求 OPTIONS 不走限流）
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# 注册全局异常处理器（异常处理器不依赖路由注册顺序，先注册也没问题）
register_exception_handlers(app)

# 注册 API 路由
app.include_router(router)


# 挂载静态文件目录（前端页面）
static_dir = os.path.join(PROJECT_ROOT, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """根路径：返回前端页面（如有），否则返回 API 信息"""
    index_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    if os.path.exists(index_path):
        from fastapi.responses import FileResponse
        return FileResponse(index_path)
    return {
        "message": "RAG 智能问答系统",
        "docs": "/docs",
        "health": "/api/health",
    }
