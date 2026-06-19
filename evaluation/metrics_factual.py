"""
========== Factual Faithfulness Metric ==========
Evaluates answer faithfulness by checking each key fact individually.
More precise than holistic 1-10 scoring.
"""

import json
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from app.utils import get_cached_llm, extract_score
from app.config import settings


FACT_CHECK_SYSTEM = (
    '你是一个严格的事实核查助手。你需要逐条判断"回答"是否包含了给定的"关键事实"，'
    '并且该事实是否能在"上下文"中找到依据。\n\n'
    '对每一条关键事实，请按照以下格式返回判断：\n'
    '事实1：[是/否/部分] - [简要说明]\n'
    '事实2：[是/否/部分] - [简要说明]\n'
    '...\n\n'
    '判断标准：\n'
    '- "是"：回答明确包含该事实，且上下文中能找到依据\n'
    '- "否"：回答未包含该事实，或包含的信息与事实矛盾\n'
    '- "部分"：回答包含了部分信息但不完整\n\n'
    '最后一行输出：总分：X/Y\n'
    '其中X为完全正确的事实数（即"是"的数量），Y为总事实数。'
)

FACT_CHECK_HUMAN = (
    '上下文：\n{context}\n\n'
    '回答：\n{answer}\n\n'
    '待核查的关键事实：\n{facts}\n\n'
    '请逐条核查：'
)

FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FACT_CHECK_SYSTEM),
    ("human", FACT_CHECK_HUMAN),
])


HALLUC_SYSTEM = (
    '你是一个严格的事实核查助手。判断以下"回答"中是否包含"上下文"中找不到依据的信息（即编造/幻觉）。\n'
    '请从1到10打分：\n'
    '1-3：严重编造，大部分信息不在上下文中\n'
    '4-6：部分编造，有些信息有依据有些没有\n'
    '7-9：基本可靠，只有少量不精确\n'
    '10：完全可靠，所有信息都在上下文中有依据\n\n'
    '只返回分数，然后换行后简要说明发现的编造内容（如有）。'
)

HALLUC_HUMAN = '上下文：\n{context}\n\n回答：\n{answer}\n\n分数及说明：'

HALLUC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HALLUC_SYSTEM),
    ("human", HALLUC_HUMAN),
])


def factual_faithfulness(answer: str, context: str, key_facts: List[str]) -> float:
    """
    Evaluate faithfulness by checking each key fact individually.
    Returns score 0.0-1.0 (proportion of facts correctly stated and supported).
    """
    if not key_facts:
        return 1.0

    llm = get_cached_llm(model=settings.judge_model, temperature=0.0)
    facts_text = '\n'.join(f'{i+1}. {f}' for i, f in enumerate(key_facts))
    prompt = FACT_CHECK_PROMPT.format(
        context=context[:4000], answer=answer[:2000], facts=facts_text,
    )
    response = llm.invoke(prompt)
    content = response.content

    # Parse "是" verdicts
    yes_count = 0
    total_count = len(key_facts)
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('事实') and '：' in line:
            verdict_part = line.split('：')[1].strip() if '：' in line else ''
            if verdict_part.startswith('是') and not verdict_part.startswith('是否'):
                yes_count += 1

    # Try summary line
    for line in content.split('\n'):
        if '总分' in line:
            m = re.search(r'(\d+)\s*/\s*(\d+)', line)
            if m:
                yes_count = int(m.group(1))
                total_count = int(m.group(2))
                break

    if total_count == 0:
        return 1.0
    return min(1.0, max(0.0, yes_count / total_count))


def hallucination_check(answer: str, context: str) -> tuple:
    """
    Check if answer contains statements unsupported by context.
    Returns (hallucination_score, details_string).
    hallucination_score: 1.0 = no hallucination, 0.0 = severe hallucination.
    """
    if not answer.strip():
        return 1.0, 'Empty answer'

    llm = get_cached_llm(model=settings.judge_model, temperature=0.0)
    prompt = HALLUC_PROMPT.format(context=context[:4000], answer=answer[:2000])
    response = llm.invoke(prompt)
    score = extract_score(response.content)
    return score, response.content.strip()
