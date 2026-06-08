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

# 创建数据目录
RUN mkdir -p /app/data/chroma_db /app/data/temp

EXPOSE 8000

# 首次启动前需要手动运行 init_db.py 初始化向量库
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
