# 🏫 RAG 智能问答系统

基于 **FastAPI + LangChain + ChromaDB + 通义千问** 的检索增强生成 (RAG) 系统，支持多策略检索、自反思验证、Web 搜索 fallback 和流式对话。

> 示例场景：北师大校园智能问答 — 但架构完全通用，切换到任意知识库只需替换 `data/source_docs/` 下的文件。

---

## 架构总览

```
                                 用户 / 前端
                       POST /api/chat  { query, ... }
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      router.py  (编排层)        │
                    │                                 │
                    │  query ──► retriever ──►        │
                    │               │                 │
                    │               ├─ rewrite_query  │
                    │               ├─ MMR search     │
                    │               ├─ rerank         │
                    │               ▼                 │
                    │         _resolve_docs()         │
                    │          KB 不足时 ──►          │
                    │         web_search.py           │
                    │         (DuckDuckGo)            │
                    │               │                 │
                    │               ▼                 │
                    │          self_rag()             │
                    │          自反思循环              │
                    │               │                 │
                    │               ▼                 │
                    │          generate()             │
                    │          qwen-plus              │
                    └──────────┬──────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────┐
                    │         ChatResponse            │
                    │  answer + sources + web_used    │
                    │  + self_rag 忠实度评分          │
                    └─────────────────────────────────┘
```

### 数据流详解

```
用户查询 query ──────────────────────────────────────────────────────┐
                                                                      │
① retriever.retrieve()                                                │
   ├─ [可选] rewrite_query(): LLM 重写查询为关键词                     │
   ├─ [可选] MMR search: 平衡相关性+多样性                             │
   └─ [可选] rerank(): DashScope gte-rerank API 精排                   │
       └─ HTTPS 调用，无需本地 GPU/内存                                 │
                                                                      │
② _resolve_docs(): Web fallback 决策                                  │
   │  max_score < 阈值? ──► web_search.py (DuckDuckGo)                │
   └─ 合并 KB + Web 结果                                               │
                                                                      │
③ self_rag_loop(): 自反思循环 (最多 N 轮)                              │
   │  generate_answer(query, docs)                                    │
   │      │                                                           │
   │  check_faithfulness(answer, context)  ← LLM 打分 0-1             │
   │      │ score < threshold?                                        │
   │      ├─ YES: generate_refinement_query() → 精炼检索 → 合并 docs  │
   │      └─ NO:  退出循环                                             │
   └─ 返回最终答案 + 忠实度评分历史                                     │
                                                                      │
④ generator.generate_answer() / generate_answer_stream()              │
   └─ qwen-plus: 基于上下文 + 问题生成最终回答                         │
```

### 模块职责

```
rag项目/
├── app/
│   ├── main.py              # FastAPI 应用入口, CORS, 静态文件挂载
│   ├── config.py            # 统一配置 (pydantic-settings, 自动加载 .env)
│   ├── router.py            # API 路由 + 流式/非流式编排 + Web fallback 决策
│   ├── schemas.py           # Pydantic 请求/响应模型 (ChatRequest/Response 等)
│   ├── database.py          # ChromaDB 连接 + DashScope Embedding 封装
│   ├── retriever.py         # 检索管道: query rewrite → MMR search → rerank
│   ├── generator.py         # LLM 生成 (同步 invoke + 流式 astream)
│   ├── self_rag.py          # 自反思循环: 生成 → 忠实度检查 → 精炼查询
│   ├── web_search.py        # Web 搜索 fallback (DuckDuckGo, 免 API Key)
│   ├── reranker.py          # Rerank API 重排序 (调用 DashScope gte-rerank)
│   ├── document_loader.py   # 文档加载 (txt/md/pdf) + 纯 Python 递归分块
│   ├── exceptions.py        # 异常体系: RAGException → Retrieval/Generation/...
│   └── error_handlers.py    # 全局异常处理器 → 统一 ErrorResponse JSON
├── scripts/
│   ├── init_db.py           # 初始化向量库 (自动扫描 data/source_docs/)
│   └── verify.py            # 8 项功能完整性验证
├── evaluation/
│   ├── metrics.py           # 4 个 LLM-as-Judge 评估指标
│   └── experiment.py        # 对比实验框架 (支持 A/B 测试)
├── data/
│   ├── source_docs/         # 知识库文件目录 (放入 .txt / .md / .pdf 自动索引)
│   └── chroma_db/           # ChromaDB 持久化数据
├── static/
│   └── index.html           # 单页聊天前端 (SSE 流式 + 选项面板)
├── requirements.txt
└── .env.example
```

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
python -m venv .venv
# 激活虚拟环境
source .venv/bin/activate    # Linux / Mac
.venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入 [DashScope](https://dashscope.aliyun.com/) API Key：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
CHAT_MODEL=qwen-plus
```

> **提示**：DashScope 新用户有免费额度，足够完成开发和测试。

### 3. 初始化向量库

```bash
python scripts/init_db.py
```

首次运行会自动将 `data/source_docs/` 下的所有 `.txt` / `.md` / `.pdf` 文件索引到 ChromaDB：

```
============================================================
初始化向量数据库
============================================================

[第 1 步] 写入内置示例文档...
   [OK] 简介.txt

[第 2 步] 扫描 .../data/source_docs 目录...
   发现 11 个文件待索引：
     - 简介.txt（3.5 KB）
     - ...

[第 4 步] 加载文档并写入向量数据库...
  -- 处理：简介.txt --
     切分：7 个片段
     已写入向量库

============================================================
初始化完成！
   处理文件数：11
   总片段数（chunks）：47
```

### 4. 运行验证

```bash
python scripts/verify.py
```

```
==================================================
  1. 配置加载
==================================================
  PROJECT_ROOT: .../rag项目
  chat_model: qwen-plus
  >>> PASS

  ... (共 8 项检查)

==================================================
  结果: 8 通过, 0 失败
==================================================
```

### 5. 启动服务

```bash
uvicorn app.main:app --reload
```

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

| 地址 | 用途 |
|------|------|
| `http://localhost:8000` | 前端聊天界面 |
| `http://localhost:8000/docs` | Swagger 交互式 API 文档 |
| `http://localhost:8000/api/health` | 健康检查 (返回向量库文档数) |

---

## 效果展示

### 前端聊天界面

```
┌─────────────────────────────────────────────┐
│  🏫 北师大智能问答助手                        │
├─────────────────────────────────────────────┤
│                                             │
│  👤 选课流程是怎样的？                        │
│                                             │
│  🤖 根据学校相关规定，选课流程一般包括         │
│     以下几个步骤：                            │
│     1. 登录教务系统查看可选课程列表            │
│     2. 在选课开放时间内进行选课操作            │
│     3. 确认选课结果并按时缴费                  │
│                                             │
│     ───────────────────────────             │
│     📎 参考来源：[1] 选课与学分.txt            │
│                [2] 简介.txt                  │
│                                             │
├─────────────────────────────────────────────┤
│ ☑ 流式输出  ☐ 重排序  ☐ 查询优化  ☐ Self-RAG │
├─────────────────────────────────────────────┤
│ [  输入你的问题...                       ] [发送]
└─────────────────────────────────────────────┘
```

### API 请求/响应示例

**基础问答：**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "违纪怎么处理？", "top_k": 5}'
```

**响应：**

```json
{
  "answer": "根据相关规定，考试违纪的处理方式包括...",
  "sources": [
    {
      "content": "第八条 对考试违纪者，视情节轻重...",
      "filename": "北师大考场规则.txt",
      "chunk_index": 3,
      "score": 0.8521,
      "source_type": "knowledge_base",
      "source_url": ""
    }
  ],
  "llm_model": "qwen-plus",
  "self_rag": null,
  "web_search_used": false
}
```

### 自反思 (Self-RAG) 效果

启用 `use_self_rag: true` 后，系统生成答案后会自动检查忠实度：

```
[INFO] Self-RAG round 1/2: faithfulness=0.40 (threshold=0.70)
       ← 忠实度不达标，自动生成精炼查询
[INFO] Refinement query: '北京师范大学 违纪处分 种类'
[INFO] Self-RAG round 2/2: faithfulness=0.90 (threshold=0.70)
       ← 达标 ✓
```

**响应包含自反思元数据：**

```json
{
  "self_rag": {
    "rounds": 2,
    "faithfulness_scores": [0.4, 0.9]
  }
}
```

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
| **Query Rewrite** | LLM 将口语化查询重写为关键词 | `use_rewrite` | `REWRITE_ENABLED` |
| **Rerank API 重排序** | DashScope gte-rerank 精排候选 | `use_reranker` | `RERANK_ENABLED` |

### 生成增强

| 特性 | 说明 | API 开关 | 服务端开关 |
|------|------|----------|-----------|
| **Self-RAG** | 生成 → 自检忠实度 → 精炼检索 → 重新生成 | `use_self_rag` | `SELF_RAG_ENABLED` |
| **Web 搜索 Fallback** | KB 不足时自动搜 DuckDuckGo 补全上下文 | `use_web_search` | `WEB_SEARCH_ENABLED` |
| **流式输出 (SSE)** | 逐 token 推送，打字机效果 | `stream` | — |

> **双层开关设计**：API 参数控制单次请求，服务端环境变量是全局总闸。管理员可随时关闭 Web 搜索或 Self-RAG 而不需改代码。

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
RERANK_ENABLED=false
RERANK_MODEL=gte-rerank
RERANK_FETCH_K=20                 # 重排序前候选池大小

# ========== Query Rewrite (可选) ==========
REWRITE_ENABLED=false
REWRITE_MODEL=qwen-turbo

# ========== Self-RAG (可选) ==========
SELF_RAG_ENABLED=false
SELF_RAG_MAX_ROUNDS=2
SELF_RAG_FAITHFULNESS_THRESHOLD=0.7

# ========== Web 搜索 Fallback (可选) ==========
WEB_SEARCH_ENABLED=false
WEB_SEARCH_FALLBACK_THRESHOLD=0.3  # KB 最高分低于此值触发 Web 搜索
WEB_SEARCH_NUM_RESULTS=5
```

---

## API 接口速查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 (返回向量库文档数) |
| `POST` | `/api/chat` | 核心问答 (支持流式 SSE, 可选 Self-RAG / Web Fallback) |
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
| 重排序 | **BGE-Reranker-v2-m3** | CrossEncoder 交叉编码器, 子进程隔离 |
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
5. **双层开关** — 每个高级功能（Reranker / Rewrite / Self-RAG / Web Search）同时受 API 参数和服务端环境变量控制
6. **SSE 流式** — 对调用方暴露简单的 `stream: bool` 开关，内部处理检索/生成/错误的全生命周期
