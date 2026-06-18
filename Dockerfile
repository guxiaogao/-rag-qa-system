# ============================================================
# RAG 智能问答系统 Dockerfile
# 构建: docker build -t rag-qa-system .
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层加速构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY static/ ./static/
COPY data/source_docs/ ./data/source_docs/
COPY docker-entrypoint.sh /app/

# 创建数据目录
RUN mkdir -p /app/data/chroma_db /app/data/temp && \
    chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
