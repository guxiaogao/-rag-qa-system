# 🏫 北师珠智能问答助手

基于 RAG（检索增强生成）的校园智能问答系统，通过对校内多来源信息统一建立知识库并优化检索链路，系统性解决 LLM 在校园场景中的"信息缺失、召回率低、幻觉难控"三大痛点。

> **项目地址**：[https://github.com/guxiaogao/-rag-qa-system](https://github.com/guxiaogao/-rag-qa-system)

---

## 项目背景

校园场景下直接使用 LLM 查询校内信息时常面临三个核心问题：

| 痛点 | 表现 | 本系统解决方案 |
|------|------|--------------|
| 网页信息缺失或过时 | 校内规章制度分散在 PDF、内部页面中，搜索引擎无法收录 | 统一收集清洗校内文档，建立私有向量知识库 |
| 口语化表述召回率低 | 学生问"翘课翘多了咋办"，向量检索难以匹配"旷课学时处分" | 查询重写 + Rerank 精排双阶段优化 |
| LLM 幻觉难控制 | 通用大模型在不知答案时容易编造 | 强制基于检索文档回答，来源可追溯，域外问题拒答 |

---

## 核心链路

```
用户提问 → Query Rewrite（可选）→ 向量检索（text-embedding-v3 + ChromaDB HNSW）
        → qwen3-rerank 精排 → 拼接上下文注入 Prompt → qwen-turbo 流式生成
                                  ↑
               Web 搜索 Fallback（DuckDuckGo，低质量时自动补全）
```

---

## 功能特性

### 数据层

- 收集并清洗北师珠校内 HTML 页面、PDF 文件、表格数据及纯文本文档，当前共 **20 余份**
- 按 **500 字符/块** 切分，相邻块重叠 **100 字符**，避免切断关键信息
- 自研纯 Python 递归文本切分器，避免 `langchain_text_splitters` 在 Windows 上的 Rust tiktoken segfault 问题

### 检索层

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 向量化模型 | `text-embedding-v3`（DashScope） | 将文档片段转为 1024 维向量 |
| 向量数据库 | ChromaDB（嵌入式模式） | HNSW 索引 + 余弦相似度，持久化存储 |
| 精排模型 | `qwen3-rerank`（DashScope Rerank API） | 语义级精排，按 token 计费，无需本地 GPU |
| 查询重写 | `qwen-turbo` | 将口语化查询转为检索友好的关键词表达 |
| Web Fallback | DuckDuckGo | 知识库检索质量不足时自动从互联网补全 |

### 生成层

| 特性 | 说明 |
|------|------|
| 生成模型 | `qwen-turbo`（低延迟，约 1-2 秒） |
| 流式输出 | 全部请求走 SSE（Server-Sent Events），逐 token 推送，打字机效果 |
| 多轮对话 | 保留最近 3 轮对话历史，识别指代给出连贯回应 |
| 来源追溯 | 回答末尾标记参考来源文件名，Web 结果附带原始链接 |
| 域外拒答 | 知识库无法覆盖的问题（如录取分数线、学科排名）主动拒绝并引导 |

### 评估体系

基于 LLM-as-Judge 范式的四维评估框架，使用 `qwen-turbo` 作为裁判模型：

| 指标 | 英文名 | 评估对象 | 含义 |
|------|--------|----------|------|
| 忠实度 | Faithfulness | 生成质量 | 回答信息是否能在检索文档中找到依据？ |
| 答案相关性 | Answer Relevancy | 生成质量 | 回答是否直接回应了问题？ |
| 检索精度 | Context Precision | 检索质量 | 检索到的文档中有多少比例真正相关？ |
| 检索召回率 | Context Recall | 检索质量 | 标准答案所需的关键信息是否都被检索到？ |

**评测数据集**：60 道题（覆盖考试纪律、违纪处分、选课、图书馆、校园生活、医疗健康、毕业、校史、研究生等 9 个主题），包含事实型、程序型、对比型和综合推理 4 种题型，以及错别字/口语化扰动变体和域外问题。

### 检索质量优化效果（15 题 Golden Set 对比）

| 指标 | 基准（纯向量检索） | +Rerank（精排）★ | 变化 |
|------|:-----------------:|:---------------:|:----:|
| MRR（平均倒数排名） | 0.75 | 0.883 | **+17.8%** |
| MAP@5（前5平均精度） | 0.685 | 0.849 | **+23.9%** |
| NDCG@5（排序质量） | 0.379 | 0.425 | **+12.3%** |
| Hit Rate@5（前5命中率） | 0.933 | **1.000** | **达到 100%** |
| Faithfulness（忠实度） | 0.947 | 0.960 | +1.3% |
| 平均延迟 | 2.07s | 2.21s | +6.8%（几乎无感） |

> **结论**：启用 `qwen3-rerank` 精排后，所有问题的正确答案均进入前 5 个结果（Hit Rate 100%），排序精度大幅提升（MAP +24%），延迟仅增加 0.14 秒。MMR 多样性策略对本场景的事实型问答有负面影响（NDCG -13.5%），不推荐使用。

### 工程化

| 方面 | 实现 |
|------|------|
| **异常处理** | 4 层自定义异常继承体系：`RAGException` → `RetrievalException` / `GenerationException` / `DocumentProcessingException` / `VectorStoreException`，全局异常处理器统一返回 `{error_code, message, detail}` 结构化 JSON |
| **配置管理** | 基于 `pydantic-settings` 自动加载 `.env` 文件，类型校验 + 默认值，敏感信息（API Key）与代码隔离 |
| **限流保护** | `SlowAPI` 按客户端 IP 独立计数：聊天接口 60 次/分钟，检索调试接口 120 次/分钟，文档上传 10 次/分钟 |
| **优雅降级** | 查询重写失败 → 自动回退原始查询；Rerank API 失败 → 回退相似度排序；Web 搜索失败 → 静默使用 KB 结果 |
| **CI 自动校验** | GitHub Actions 三阶段流水线：PR 快速检查（~30s，无 API 消耗）→ main 分支全量集成验证（初始化向量库 + 端到端检索 + 冒烟评估）→ 手动触发全量 Benchmark |

---

## 项目结构

```
rag项目/
├── app/                        # 核心应用模块
│   ├── config.py               # 配置管理（pydantic-settings）
│   ├── database.py             # ChromaDB 连接与操作
│   ├── document_loader.py      # 文档加载与递归文本切分器
│   ├── exceptions.py           # 4 层自定义异常体系
│   ├── error_handlers.py       # FastAPI 全局异常处理器
│   ├── generator.py            # LLM 回答生成（同步 + 流式）
│   ├── main.py                 # FastAPI 应用入口
│   ├── reranker.py             # DashScope Rerank API 封装
│   ├── retriever.py            # 检索管线（向量检索 + 重写 + 精排）
│   ├── router.py               # RESTful API 路由 + Web Fallback 决策
│   ├── schemas.py              # Pydantic 请求/响应模型
│   ├── utils.py                # 工具函数（LLM 缓存、分数提取）
│   └── web_search.py           # DuckDuckGo Web 搜索
├── scripts/                    # 运维脚本
│   ├── init_db.py              # 向量库初始化（增量/全量）
│   ├── verify.py               # 全量功能验证（调真实 API）
│   └── ci_verify.py            # CI 快速检查（无 API 消耗）
├── evaluation/                 # 评估模块
│   ├── metrics.py              # 4 维 LLM-as-Judge 评估指标
│   ├── metrics_factual.py      # 事实忠实度 + 幻觉检测
│   ├── metrics_retrieval.py    # 检索质量指标（MRR/NDCG/Hit Rate/MAP）
│   ├── experiment.py           # 对比实验框架 + Bootstrap 置信区间
│   ├── runner.py               # 统一 CLI 入口（benchmark/golden/robustness/oos）
│   ├── param_sweep.py          # 参数网格搜索
│   ├── _build_data.py          # 测试数据集生成
│   └── results/                # 评估结果存档
├── data/
│   ├── source_docs/            # 源文档（.txt/.md/.pdf）
│   └── chroma_db/              # ChromaDB 持久化数据（gitignore）
├── static/
│   └── index.html              # 前端聊天界面
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── requirements.txt
├── .env.example                # 环境变量模板
└── README.md
```

---

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 2. 本地运行

```bash
pip install -r requirements.txt
python scripts/init_db.py        # 初始化向量库
uvicorn app.main:app --reload    # 启动服务（http://localhost:8000）
```

### 3. Docker 部署

```bash
docker-compose up -d
# 首次启动自动初始化向量库，无需手动执行 init_db.py
```

### 4. 运行评估

```bash
python -m evaluation.runner benchmark              # 基准测试
python -m evaluation.runner benchmark --full        # 多配置对比
python -m evaluation.runner benchmark --quick       # 快速冒烟
python -m evaluation.runner golden                  # Golden 回归测试
python -m evaluation.runner robustness              # 鲁棒性测试
python -m evaluation.runner oos                     # 域外拒答测试
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查，返回向量库文档数 |
| `POST` | `/api/chat` | 核心问答接口（SSE 流式输出） |
| `POST` | `/api/search` | 仅检索不生成（调试用） |
| `GET` | `/api/documents` | 列出已索引文档及分块数 |
| `POST` | `/api/documents/upload` | 上传并自动索引文档 |
| `DELETE` | `/api/documents/{filename}` | 删除指定文档的索引 |

启动服务后访问 `http://localhost:8000/docs` 查看完整交互式文档。

---

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| Web 框架 | FastAPI（异步、OpenAPI、Pydantic 校验） |
| LLM & Embedding | DashScope 通义千问（qwen-turbo / text-embedding-v3 / qwen3-rerank） |
| 向量数据库 | ChromaDB（嵌入式模式，HNSW + SQLite 持久化） |
| LLM 编排 | LangChain（Document / Prompt 抽象、Chroma 集成） |
| PDF 解析 | pypdf |
| Web 搜索 | DuckDuckGo（免 API Key，零配置 Fallback） |
| 限流 | SlowAPI（按 IP 独立计数） |
| 评估 | LLM-as-Judge + Pandas 对比分析 |
| 配置管理 | pydantic-settings |
| 部署 | Docker + Docker Compose，目标平台阿里云 ECS |
| CI | GitHub Actions（PR 快速检查 / main 全量验证 / 手动 Benchmark） |

---

## 配置参考

完整配置项见 `.env.example`，核心参数：

```env
# 模型
CHAT_MODEL=qwen-turbo               # 答案生成（低延迟）
EMBEDDING_MODEL=text-embedding-v3   # 文本向量化
JUDGE_MODEL=qwen-turbo              # 评估裁判

# 检索
CHUNK_SIZE=500                      # 分块大小（字符数）
CHUNK_OVERLAP=100                   # 分块重叠
TOP_K=5                             # 返回片段数

# 精排
RERANK_ENABLED=true                 # 强烈建议开启
RERANK_MODEL=qwen3-rerank
RERANK_FETCH_K=10                   # 精排前候选池大小

# Web Fallback
WEB_SEARCH_ENABLED=false            # 生产环境建议关闭（校内数据公网不可检索）
WEB_SEARCH_FALLBACK_THRESHOLD=0.3
```
