# 🏫 RAG 智能问答系统

### Web 搜索 Fallback 效果

当知识库中找不到相关内容时，自动搜索互联网补充：

```
[INFO] Web fallback 触发：KB max_score=0.180 < threshold=0.300，
       合并 5 条 web 结果
```

**响应中标记来源类型：**

```json
{
  "web_search_used": true,
  "sources": [
    {
      "filename": "Web: Transformer (deep learning architecture)",
      "source_type": "web",
      "source_url": "https://en.wikipedia.org/wiki/Transformer",
      "score": 0.5
    },
    {
      "filename": "知识库文档.pdf",
      "source_type": "knowledge_base",
      "score": 0.18
    }
  ]
}
```

---

## 功能特性

### 检索管道

| 特性 | 说明 | API 开关 | 服务端开关 |
|------|------|----------|-----------|
| **向量相似度检索** | HNSW 索引 + cosine 距离 | 默认启用 | — |
| **MMR 检索** | 最大边际相关性，平衡相关性和多样性 | `use_mmr` | — |
| **Query Rewrite** | LLM 将口语化查询重写为关键词 | `use_rewrite` | `REWRITE_ENABLED` (默认开启) |
| **Rerank API 重排序** | DashScope gte-rerank 精排候选 | `use_reranker` | `RERANK_ENABLED` (默认开启) |

### 生成增强

| 特性 | 说明 | API 开关 | 服务端开关 |
|------|------|----------|-----------|
| **Web 搜索 Fallback** | KB 不足时自动搜 DuckDuckGo 补全上下文 | — | `WEB_SEARCH_ENABLED` |
| **流式输出 (SSE)** | 逐 token 推送，打字机效果 | `stream` | — |

> **双层开关设计**：API 参数控制单次请求（Reranker / Rewrite），服务端环境变量是全局总闸。Web 搜索仅由服务端环境变量 + 自动判断控制，无需用户手动开关。

### 评估体系

`evaluation/metrics.py` 实现 4 个 LLM-as-Judge 评估指标：

| 指标 | 含义 | 评估对象 |
|------|------|----------|
| **Faithfulness** | 答案中的信息都能在检索文档中找到依据吗？ | 生成质量 |
| **Answer Relevancy** | 答案是否直接回应了问题？ | 生成质量 |
| **Context Precision** | 检索到的文档中有多少是真正相关的？ | 检索质量 |
| **Context Recall** | 标准答案所需的关键信息是否都被检索到了？ | 检索质量 |

`evaluation/experiment.py` 提供对比实验框架，可一键跑多组参数配置 (chunk_size / top_k / MMR / Reranker)，输出 DataFrame 对比表。

### 异常处理体系

每个模块异常向上传播，由 `error_handlers.py` 统一捕获并返回结构化 JSON：

```json
{
  "error_code": "RETRIEVAL_ERROR",
  "message": "文档检索失败，请稍后重试",
  "detail": "ChromaDB 连接超时"
}
```

异常层级：`RAGException` → `RetrievalException` / `GenerationException` / `DocumentProcessingException` / `VectorStoreException`

---

## 配置参考

完整配置项见 [.env.example](.env.example)，以下为核心参数：

```env
# ========== 模型配置 ==========
DASHSCOPE_API_KEY=sk-...
CHAT_MODEL=qwen-plus              # 答案生成模型
JUDGE_MODEL=qwen-turbo            # 评估/重写/裁判模型 (便宜)
EMBEDDING_MODEL=text-embedding-v2 # 向量化模型

# ========== 检索参数 ==========
TOP_K=5                           # 默认返回片段数
CHUNK_SIZE=500                    # 文档分块大小 (字符数)
CHUNK_OVERLAP=100                 # 分块之间重叠量

# ========== 重排序 (可选, 走 API, 极低成本) ==========
RERANK_ENABLED=true              # 建议开启，极低成本精排
RERANK_MODEL=gte-rerank
RERANK_FETCH_K=20                 # 重排序前候选池大小

# ========== Query Rewrite (可选) ==========
REWRITE_ENABLED=true              # 建议开启，优化检索关键词
REWRITE_MODEL=qwen-turbo

# ========== Web 搜索 Fallback ==========
WEB_SEARCH_ENABLED=true
WEB_SEARCH_FALLBACK_THRESHOLD=0.3  # KB 最高分低于此值触发 Web 搜索
WEB_SEARCH_NUM_RESULTS=5
```

---

## API 接口速查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 (返回向量库文档数) |
| `POST` | `/api/chat` | 核心问答 (支持流式 SSE，自动 Web Fallback) |
| `POST` | `/api/search` | 仅检索不生成 (调试用) |
| `GET` | `/api/documents` | 列出所有已索引文档及分块数 |
| `POST` | `/api/documents/upload` | 上传并自动索引文档 (.txt/.md/.pdf) |
| `DELETE` | `/api/documents/{filename}` | 删除指定文档的所有索引 |

完整交互式文档：启动服务后访问 `http://localhost:8000/docs`

---

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| Web 框架 | **FastAPI** | 异步支持、自动 OpenAPI、Pydantic 校验 |
| LLM & Embedding | **DashScope 通义千问** | OpenAI 兼容接口, 中文效果优秀 |
| 向量数据库 | **ChromaDB** | HNSW 索引, cosine 距离, 持久化存储 |
| LLM 编排 | **LangChain** | Document / Prompt 抽象, Chroma 集成 |
| 重排序 | **DashScope gte-rerank API** | 云端重排序, 按 token 计费, 无需本地 GPU |
| PDF 解析 | **pypdf** | 纯 Python PDF 文本提取 |
| Web 搜索 | **duckduckgo-search** | 免 API Key, 零配置 fallback |
| 评估 | **LLM-as-Judge** | 4 指标自评估, Pandas 对比分析 |

---

## 自定义知识库

切换到你的领域只需两步：

```bash
# 1. 清空旧文档，放入你的文件
rm data/source_docs/*
cp /path/to/your/docs/*.pdf  data/source_docs/
cp /path/to/your/docs/*.txt  data/source_docs/
cp /path/to/your/docs/*.md   data/source_docs/

# 2. 重建向量索引
python scripts/init_db.py
```

**支持格式**：`.txt` / `.md` / `.pdf`

**分块策略**：默认每 500 字符一块，相邻块重叠 100 字符。可在 `.env` 中调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`。

---

## 项目特色

1. **模块解耦** — retriever / generator / reranker 各自独立，替换任一模块不影响其余
2. **优雅降级** — Query Rewrite 失败 → 用原始查询；Web Search 失败 → 继续用 KB 结果；Faithfulness Check 失败 → 保守返回 1.0
3. **API 优先** — 重排序走 DashScope gte-rerank API，无需本地模型，彻底消除 PyTorch/PyArrow DLL 冲突，Docker 镜像体积大幅缩减
4. **纯 Python 分块** — 自主实现 `RecursiveTextSplitter`，避免 `langchain_text_splitters` 在部分 Windows 环境的 Rust tiktoken segfault 问题
5. **双层开关** — Reranker / Rewrite 同时受 API 参数和服务端环境变量控制；Web 搜索由服务端开关 + 相关性阈值自动判断，无需用户介入
6. **SSE 流式** — 对调用方暴露简单的 `stream: bool` 开关，内部处理检索/生成/错误的全生命周期
