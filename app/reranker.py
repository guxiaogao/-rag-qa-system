"""
========== 重排序模块 ==========
通过 DashScope Rerank API (qwen3-rerank) 对第一阶段检索结果进行精排。
使用 DashScope 原生 Rerank API（非 OpenAI 兼容接口路径）。

优势相较于本地 CrossEncoder：
- 无需本地加载模型（省 2GB+ 内存，Docker 镜像瘦身 ~3GB）
- 无需下载 HuggingFace 模型文件（告别首次启动 2-5 分钟等待）
- 无需子进程隔离（彻底解决 PyTorch / PyArrow 在 Windows 上的 DLL 冲突）
- 按 token 计费，成本极低（约 ¥0.0007 / 千 tokens）
- 效果与 BAAI/bge-reranker-v2-m3 同级别

API 文档：https://help.aliyun.com/zh/model-studio/reranker
"""

import json
import urllib.request
import urllib.error
from typing import List

from langchain_core.documents import Document as LCDocument

from app.config import settings

# DashScope 原生 Rerank API 地址
# qwen3-rerank 使用原生接口（OpenAI 兼容模式不支持 rerank 路径）
_RERANK_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

# 重试参数
_MAX_RETRIES = 0           # 不重试：失败由 retriever.py 降级逻辑兜底
_API_TIMEOUT = 3           # 3s 超时（正常 Rerank API 200-400ms 返回）


def _call_rerank_api(request_body: dict, headers: dict) -> dict:
    """
    调用 DashScope Rerank API。

    无重试：失败直接抛出，由 retriever.py 的降级逻辑兜底
    （用原始检索结果替代 Rerank 结果，用户无感知）。
    """
    try:
        req = urllib.request.Request(
            _RERANK_API_URL,
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(
            f"Rerank API 返回 HTTP {e.code}: {error_body[:300]}"
        )

    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接 Rerank API: {e.reason}")

    except Exception as e:
        raise RuntimeError(
            f"Rerank API 调用失败 ({type(e).__name__}): {str(e)}"
        )


def rerank(
    query: str,
    docs: List[LCDocument],
    top_k: int,
    model_name: str = None,
) -> List[LCDocument]:
    """
    通过 DashScope Rerank API 对候选文档重排序，返回最相关的 top_k 个。

    参数：
        query:      用户问题
        docs:       第一阶段检索的候选文档列表
        top_k:      最终返回的文档数量
        model_name: 可选，覆盖默认模型（默认使用 settings.rerank_model 即 qwen3-rerank）

    返回：
        重排序后的文档列表（长度 ≤ top_k），每个 doc.metadata 中含有 "rerank_score"

    注意：
        - 原始 embedding 分数保留在 doc.metadata["score"] 中
        - 如果候选数 ≤ top_k，不做截断，但仍打分和排序
    """
    if not docs:
        return []

    model = model_name or settings.rerank_model
    doc_texts = [doc.page_content for doc in docs]

    request_body = {
        "model": model,
        "input": {
            "query": query,
            "documents": doc_texts,
        },
        "parameters": {
            "top_n": min(top_k, len(docs)),
            "return_documents": False,
        },
    }

    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "Content-Type": "application/json",
    }

    # 带重试的 API 调用
    response_data = _call_rerank_api(request_body, headers)

    # API 返回错误时，响应中包含非空 "code" 字段
    if "code" in response_data and response_data.get("code"):
        error_msg = response_data.get("message", "未知错误")
        raise RuntimeError(f"Rerank API 错误: {error_msg}")

    results = response_data.get("output", {}).get("results", [])

    # 如果 API 返回空结果（异常情况），静默降级：
    # 直接返回原始文档列表（按相似度排序后的 top_k），保留已有上下文
    if not results:
        return sorted(docs, key=lambda d: d.metadata.get("score", 0.0), reverse=True)[:top_k]

    # 将重排序分数写入各文档 metadata
    for item in results:
        idx = item.get("index", -1)
        score = item.get("relevance_score", 0.0)
        if 0 <= idx < len(docs):
            docs[idx].metadata["rerank_score"] = round(score, 4)

    # 按重排序分数降序排列，取 top_k
    ranked = sorted(
        [item for item in results if 0 <= item.get("index", -1) < len(docs)],
        key=lambda x: x.get("relevance_score", 0.0),
        reverse=True,
    )
    result = [docs[item["index"]] for item in ranked[:top_k]]

    return result
