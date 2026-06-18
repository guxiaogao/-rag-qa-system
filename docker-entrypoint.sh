#!/bin/bash
# ============================================================
# RAG 智能问答系统 — Docker 启动入口
# 功能：
#   1. 首次启动自动初始化向量数据库（如 chroma_db 为空）
#   2. 启动 uvicorn 服务
#
# 注意：本项目的耗时任务均为 I/O 密集型（DashScope / ChromaDB），
# asyncio 单进程事件循环已能高效处理并发。多 worker 不适合本项目，
# 原因：① slowapi 限流计数器是进程內存的，多 worker 会稀释限流；
#       ② ChromaDB 底层 sqlite3 多进程写入存在锁冲突。
# ============================================================
set -e

DB_FILE="/app/data/chroma_db/chroma.sqlite3"

if [ ! -f "$DB_FILE" ]; then
    echo "============================================================"
    echo "  首次启动：正在初始化向量数据库..."
    echo "============================================================"
    python scripts/init_db.py
    echo ""
    echo "============================================================"
    echo "  初始化完成，启动服务..."
    echo "============================================================"
else
    echo "向量数据库已存在，跳过初始化"
fi

# 日志级别：默认 warning（生产环境减少噪音）。
# uvicorn --log-level 控制 HTTP 访问日志，RAG_LOG_LEVEL 控制业务日志。
# 开发环境可设为 info 或 debug。
LOG_LEVEL="${LOG_LEVEL:-warning}"
export RAG_LOG_LEVEL="${RAG_LOG_LEVEL:-$LOG_LEVEL}"

# uvicon 运行时参数
WORKERS="${UVICORN_WORKERS:-1}"              # 默认 1（I/O 密集型不需多 worker）
LIMIT_CONCURRENCY="${UVICORN_LIMIT_CONCURRENCY:-100}"  # 最大并发连接数
TIMEOUT_KEEP_ALIVE="${UVICORN_TIMEOUT_KEEP_ALIVE:-5}"   # keep-alive 超时秒数

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${WORKERS}" \
    --limit-concurrency "${LIMIT_CONCURRENCY}" \
    --timeout-keep-alive "${TIMEOUT_KEEP_ALIVE}" \
    --log-level "${LOG_LEVEL}"
