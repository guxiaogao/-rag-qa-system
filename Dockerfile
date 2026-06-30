# ============================================================
# RAG 智能问答系统 Dockerfile
# 构建: docker build -t rag-qa-system .
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# 替换 apt 源为阿里云镜像（国内 ECS 大幅加速）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层加速构建
COPY requirements.txt .
# pip 使用阿里云镜像加速（国内 ECS 提速 10 倍+）
RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    -r requirements.txt

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
