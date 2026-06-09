"""
========== 评估指标模块 ==========
实现 RAG 系统的 4 个核心评估指标：
1. Faithfulness（忠实度）：答案是否基于检索到的上下文，没有编造
2. Answer Relevancy（答案相关性）：答案是否针对问题
3. Context Precision（检索精度）：检索到的文档中有多少是相关的
4. Context Recall（检索召回）：需要的文档是否都被检索到了

所有指标使用 LLM-as-Judge 范式，用通义千问做裁判。
"""

import json
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.utils import get_cached_llm


def get_judge_llm(temperature: float = 0.0):
    """
    获取（缓存）裁判 LLM 实例。
    用 qwen-turbo（便宜）做评估，temperature=0 保证结果稳定。
    同一 (model, temperature) 组合全局复用，避免重复创建 HTTP 客户端。
    """
    return get_cached_llm(model=settings.judge_model, temperature=temperature)


from app.utils import extract_score as _extract_score  # 共享工具函数，与 app/self_rag.py 同源


# ========== 指标 1：Faithfulness（忠实度）==========

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个评估助手。你需要判断以下"回答"是否忠实于给定的"上下文"。
忠实意味着：回答中的所有信息都能在上下文中找到依据，没有编造。
不忠实意味着：回答中包含上下文中没有的信息，或者与上下文矛盾。
请从 1 到 10 打分：
1-3：严重编造，大部分信息不在上下文中
4-6：部分编造，有些信息可以找到依据，有些不能
7-9：基本忠实，只有少量不精确之处
10：完全忠实，所有信息都有上下文依据

只返回分数，不要多余的文字。"""),
    ("human", "上下文：\n{context}\n\n回答：\n{answer}\n\n分数："),
])


def faithfulness(answer: str, context: str) -> float:
    """评估回答是否忠实于检索到的上下文"""
    llm = get_judge_llm()
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = llm.invoke(prompt)
    return _extract_score(response.content)


# ========== 指标 2：Answer Relevancy（答案相关性）==========

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个评估助手。你需要判断以下"回答"是否与"问题"相关。

相关意味着：回答直接针对问题，提供了问题所需要的信息。
不相关意味着：回答跑题了，或者答非所问。
请从 1 到 10 打分：
1-3：完全不相关
4-6：部分相关，但没切中要点
7-9：比较相关
10：非常相关，完美回答了问题
只返回分数，不要多余的文字。"""),
    ("human", "问题：{question}\n\n回答：{answer}\n\n分数："),
])


def answer_relevancy(question: str, answer: str) -> float:
    """评估回答是否与问题相关"""
    llm = get_judge_llm()
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    response = llm.invoke(prompt)
    return _extract_score(response.content)


# ========== 指标 3：Context Precision（检索精度）==========

PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个评估助手。你需要判断以下"检索结果"中的每个片段是否与"问题"相关。

请对每个片段逐一判断，返回每个片段是否相关（是/否），以及相关的比例。

格式要求：
片段1：是/否
片段2：是/否
...
相关比例：X/Y"""),
    ("human", "问题：{question}\n\n检索到的文档片段：\n{context}\n\n请依次判断每个片段是否相关："),
])


def context_precision(question: str, context_chunks: List[str]) -> float:
    """
    评估检索精度。
    即检索到的 top-k 个片段中有多少比例是真正相关的。
    """
    llm = get_judge_llm()
    context_text = "\n---\n".join(
        [f"片段{i+1}：{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    prompt = PRECISION_PROMPT.format(question=question, context=context_text)
    response = llm.invoke(prompt)
    content = response.content

    # 计算相关比例
    relevant_count = 0
    total_count = 0
    for line in content.split("\n"):
        if "片段" in line and ("是" in line or "否" in line):
            total_count += 1
            if "是" in line and "否" not in line:
                relevant_count += 1

    if total_count == 0:
        return 0.5
    return relevant_count / total_count


# ========== 指标 4：Context Recall（检索召回）==========

RECALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个评估助手。你需要判断"检索到的文档片段"是否覆盖了回答"标准答案"所需的关键信息。

关键问题：基于标准答案中的信息，检索结果是否包含了足够的内容来得出这个答案？

请从 1 到 10 打分：
1-3：严重遗漏，大部分必要信息未检索到
4-6：部分遗漏，一些关键信息缺失
7-9：基本覆盖，只有少量信息缺失
10：完全覆盖，所有必要信息都已检索到

只返回分数，不要多余的文字。"""),
    ("human", "标准答案：{golden_answer}\n\n检索到的文档片段：\n{context}\n\n分数："),
])


def context_recall(golden_answer: str, context: str) -> float:
    """评估检索到的内容是否覆盖了标准答案所需的信息"""
    llm = get_judge_llm()
    prompt = RECALL_PROMPT.format(golden_answer=golden_answer, context=context)
    response = llm.invoke(prompt)
    return _extract_score(response.content)
