"""
========== Web 搜索模块 ==========
作为知识库检索的后备方案：当向量库结果质量不足时，从公共搜索引擎
获取网页摘要补充上下文。

选择 DuckDuckGo 的理由：
- 免 API Key，零配置即可启动
- duckduckgo-search 库封装良好
- 作为 fallback，非主路径，无需付费 API

返回值与 retriever.py 完全同构（List[LCDocument]），
上游 router.py 不需要区分"这段上下文来自 KB 还是 Web"。

边界情况：
- 库未安装 → 静默降级，返回空列表
- 网络异常 / 搜索失败 → 静默降级，返回空列表
  不抛异常，因为这是 fallback 路径，失败不应阻塞主流程
"""

import logging
from typing import List

from langchain_core.documents import Document as LCDocument
from app.config import settings

logger = logging.getLogger("rag.web_search")


def web_search(query: str, num_results: int = None) -> List[LCDocument]:
    """
    搜索网页，返回 LangChain Document 列表。

    参数：
        query:       搜索查询（用户原始问题）
        num_results: 返回结果数，默认使用 settings.web_search_num_results

    返回：
        LCDocument 列表，每个包含：
          page_content  → 网页摘要文本
          metadata:
            filename     → "Web: {网页标题}"
            source_url   → 网页原始链接
            source_type  → "web"
            score        → 0.5（中性分）
            chunk_index  → 0

    为什么 score=0.5？
        搜索引擎的排序不代表与问题的语义相关性等同向量检索，
        用中性分是诚实的表态，避免前端误以为"web 结果同样可靠"。
    """
    if num_results is None:
        num_results = settings.web_search_num_results

    try:
        from duckduckgo_search import DDGS

        results: List[LCDocument] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append(LCDocument(
                    page_content=r["body"],
                    metadata={
                        "filename": f"Web: {r['title']}",
                        "source_url": r.get("href", ""),
                        "source_type": "web",
                        "score": 0.5,
                        "chunk_index": 0,
                    },
                ))

        logger.info(
            "Web search: '%s' → %d results",
            query[:80], len(results),
        )
        return results

    except ImportError:
        logger.error(
            "duckduckgo-search 未安装，Web 搜索不可用。"
            "请执行: pip install duckduckgo-search"
        )
        return []

    except Exception as e:
        logger.warning(
            "Web 搜索失败 (%s: %s)，降级使用知识库结果",
            type(e).__name__, str(e)[:120],
        )
        return []
