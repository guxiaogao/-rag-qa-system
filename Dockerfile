# ============================================================
# RAG 智能问答系统 Dockerfile
# 基于 FastAPI + LangChain + ChromaDB + 通义千问
# 所有重计算（Embedding、Rerank、Chat）走 DashScope API，无需 GPU
# ============================================================

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（仅保留必要的运行时库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .

# 安装 Python 依赖（轻量，全部走 PyPI 预编译 wheel）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建非 root 用户运行（安全最佳实践）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ChromaDB 向量数据库持久化目录
VOLUME ["/app/data/chroma_db"]

# FastAPI 默认端口
EXPOSE 8000

# 启动服务
# --host 0.0.0.0 允许容器外部访问
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
