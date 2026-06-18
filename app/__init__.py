"""
========== RAG 应用初始化 ==========
在任何 rag 子模块被导入之前，强制接管项目自身的日志配置。

为什么不用 basicConfig？
- Uvicorn 启动时会先于 main.py 给根 Logger 绑定 handler，basicConfig 碰到已有 handler
  的根 Logger 会被静默跳过（这是 Python logging 模块的"一次性"设计）。
- 子模块（retriever.py, generator.py 等）的 logger.info() 调在函数体内，不在模块级，
  所以不涉及"日志产生早于配置"的时序问题。

为什么用 dictConfig 而非 basicConfig？
- dictConfig 不受"根 Logger 已有 handler 就跳过"的限制，可随时增量修改日志系统。
- disable_existing_loggers=False：保留 Uvicorn 自带的 uvicorn.access / uvicorn.error
  等 logger 不受影响。
- propagate=False：避免同一条日志既走 rag 的 handler 又传播到根 Logger（uvicorn 的
  handler），造成控制台双份输出。

在非 Uvicorn 环境下（如 python scripts/init_db.py）：
  scripts 通过 sys.path.insert 导入 app.*，__init__.py 最先被加载，此时根 Logger
  还没有任何 handler → dictConfig 正常生效，rag 日志有完整格式输出。
"""

import os
import logging.config
from pathlib import Path

# pydantic-settings 的 Settings() 实例在 config.py 中创建，但 __init__.py 在它之前运行。
# 为了在 dictConfig 时能读到 .env 中的 RAG_LOG_LEVEL，这里手动加载一次 .env。
# python-dotenv 是 pydantic-settings 的传递依赖，已随 requirements.txt 自动安装。
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    _load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv 不可用时静默跳过，RAG_LOG_LEVEL 回退到默认值

# 业务日志级别：从环境变量 RAG_LOG_LEVEL 读取，默认 INFO。
# 开发环境设为 debug/info；生产环境设为 warning 以减少噪音。
# 与 uvicorn 的 --log-level 独立控制（互不干扰），但建议保持一致。
_rag_log_level = os.getenv("RAG_LOG_LEVEL", "INFO").upper()

_RAG_LOGGING_CONFIG = {
    "version": 1,
    # 关键：不关闭任何已有 logger。Uvicorn 启动时已用 dictConfig 配置了
    # uvicorn.access / uvicorn.error / root，设为 False 只追加不覆盖。
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(name)s] %(levelname)-8s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        # 为 rag 及所有子 logger（rag.retriever, rag.router, ...）绑定 handler。
        # 子 logger 默认从父级继承 handler，无需逐个列举。
        "rag": {
            "level": _rag_log_level,
            "handlers": ["console"],
            # 不向上传播到 root。Uvicorn 已给 root 配置了它自己的彩色格式，
            # 传播上去会导致 rag 日志出现第二次（且格式不一致）。
            "propagate": False,
        },
    },
}

# 在 import 任何 rag 子模块之前执行
logging.config.dictConfig(_RAG_LOGGING_CONFIG)
