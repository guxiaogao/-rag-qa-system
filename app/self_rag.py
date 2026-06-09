"""
========== Self-RAG 自我反思模块 ==========
生成答案后让 LLM 自检：每个声称的事实都能在检索结果中找到支撑吗？
如果不能 → 自动生成精炼查询 → 重新检索 → 重新生成。

这是 2024-2025 年 RAG 领域最核心的研究方向之一：让 RAG 系统具备自我纠错能力。

流程:
    generate_answer(query, docs)
        ↓
    check_faithfulness(answer, context)  ← LLM-as-Judge 打分 (0-1)
        ↓ score < threshold?
    generate_refinement_query()          ← 找出无据可查的内容，生成新检索词
        ↓
    retrieve(refined_query) → 合并 docs
        ↓
    regenerate → 重新打分 → 最多 max_rounds 轮
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.generator import generate_answer, format_context
from app.utils import get_cached_llm

logger = logging.getLogger("rag.self_rag")

# ========== 忠实度检查 Prompt ==========
# 与 evaluation/metrics.py 的 FAITHFULNESS_PROMPT 保持一致

FAITHFULNESS_CHECK_PROMPT = ChatPromptTemplate.from_messages([
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

# ========== 精炼查询生成 Prompt ==========

REFINEMENT_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个查询优化助手。系统生成的答案中存在部分内容在检索文档中找不到依据（幻觉）。

请基于以下信息，生成一个新的检索查询，用于找到能够支撑答案中薄弱部分的相关文档。

要求：
1. 找出答案中那些可能缺乏上下文支撑的关键主张
2. 生成一个针对这些薄弱点的、简洁的检索查询
3. 只输出检索查询文本，不要任何解释或额外内容"""),
    ("human", "原始问题：{original_query}\n\n检索到的文档：\n{context}\n\n生成的答案：\n{answer}\n\n新的检索查询："),
])


from app.utils import extract_score as _extract_score  # 共享工具函数，与 evaluation/metrics.py 同源


def check_faithfulness(answer: str, context: str) -> tuple[float, bool]:
    """
    用廉价 LLM 评估答案是否忠实于上下文。

    参数：
        answer:  LLM 生成的回答
        context: 检索到的文档上下文（format_context() 的输出）

    返回：
        (faithfulness_score, degraded)
        - faithfulness_score: 0-1 之间的忠实度分数，>0.7 表示基本忠实
        - degraded: True 表示裁判 LLM 调用失败，分数不可靠（保守返回 1.0）
        失败时返回 (1.0, True)：保守策略不触发精炼，但通过 degraded 标记通知调用方
    """
    if not answer or not context:
        return 1.0, False

    try:
        llm = get_cached_llm(model=settings.judge_model, temperature=0.0)
        prompt = FAITHFULNESS_CHECK_PROMPT.format(context=context, answer=answer)
        response = llm.invoke(prompt)
        score = _extract_score(response.content)
        logger.info("Faithfulness check: %s → score=%.2f", answer[:60], score)
        return score, False
    except Exception as e:
        logger.warning(
            "Faithfulness check 失败 (%s: %s)，默认返回 1.0（不触发精炼），标记 degraded=True",
            type(e).__name__, str(e)[:120],
        )
        return 1.0, True


def generate_refinement_query(
    answer: str,
    docs: List[LCDocument],
    original_query: str,
) -> Optional[str]:
    """
    基于幻觉检测结果，生成新的检索查询来补充缺失信息。

    参数：
        answer:         当前答案（可能存在幻觉）
        docs:           当前检索到的文档
        original_query: 用户原始问题

    返回：
        新的检索查询字符串，或 None（生成失败时）
    """
    if not answer:
        return None

    try:
        llm = get_cached_llm(model=settings.judge_model, temperature=0.0)
        context = format_context(docs)
        prompt = REFINEMENT_QUERY_PROMPT.format(
            original_query=original_query,
            context=context,
            answer=answer,
        )
        response = llm.invoke(prompt)
        refined = response.content.strip()
        if refined:
            logger.info("Refinement query: '%s'", refined[:80])
            return refined
    except Exception as e:
        logger.warning("Refinement query 生成失败 (%s)", type(e).__name__)

    return None


def self_rag_loop(
    query: str,
    docs: List[LCDocument],
    temperature: float = None,
    max_rounds: int = 2,
    faithfulness_threshold: float = 0.7,
    refine_top_k: int = 3,
    conversation_history: list[dict] = None,
) -> dict:
    """
    Self-RAG 主循环。

    流程：
    for round in 1..max_rounds:
        answer = generate_answer(query, docs)
        score, degraded = check_faithfulness(answer, context)
        if score >= threshold: break
        refined = generate_refinement_query(answer, docs, query)
        if not refined: break
        new_docs = retrieve(refined, top_k=refine_top_k)
        docs = docs + new_docs

    参数：
        query:                  用户原始问题
        docs:                   初始检索到的文档
        temperature:            LLM 生成温度
        max_rounds:             最大精炼轮次
        faithfulness_threshold: 忠实度阈值，低于此分数触发精炼
        refine_top_k:           精炼检索的 top_k
        conversation_history:   多轮对话历史（仅在首轮生成时使用）

    返回：
        {
            "answer": str,                   # 最终答案
            "docs": List[LCDocument],        # 最终文档列表（可能合并了精炼文档）
            "rounds": int,                   # 实际精炼轮次
            "faithfulness_scores": [float],   # 每轮的忠实度分数
            "degraded": bool,                # 是否有任何裁判 LLM 调用失败
        }
    """
    from app.retriever import retrieve

    if max_rounds < 1:
        raise ValueError("max_rounds 必须 >= 1")

    current_docs = docs
    answer = ""  # 防御性初始化：防止循环未执行时的 UnboundLocalError
    scores = []
    degraded = False  # 跟踪是否有裁判 LLM 降级

    for round_num in range(max_rounds):
        # 生成答案（仅在首轮使用对话历史）
        answer = generate_answer(
            query=query,
            docs=current_docs,
            temperature=temperature,
            conversation_history=conversation_history if round_num == 0 else None,
        )

        # 检查忠实度
        context = format_context(current_docs)
        score, round_degraded = check_faithfulness(answer, context)
        scores.append(round(score, 4))
        if round_degraded:
            degraded = True

        logger.info(
            "Self-RAG round %d/%d: faithfulness=%.2f (threshold=%.2f)%s",
            round_num + 1, max_rounds, score, faithfulness_threshold,
            " [degraded]" if round_degraded else "",
        )

        if score >= faithfulness_threshold:
            # 忠实度达标，返回当前答案
            return {
                "answer": answer,
                "docs": current_docs,
                "rounds": round_num + 1,
                "faithfulness_scores": scores,
                "degraded": degraded,
            }

        # 忠实度不达标，尝试精炼
        if round_num < max_rounds - 1:
            refined_query = generate_refinement_query(
                answer=answer,
                docs=current_docs,
                original_query=query,
            )
            if not refined_query:
                # 精炼查询生成失败，终止循环
                break

            # 用精炼查询重新检索
            try:
                new_docs = retrieve(
                    query=refined_query,
                    top_k=refine_top_k,
                )
                # 合并文档（去重）
                existing_contents = {d.page_content for d in current_docs}
                for d in new_docs:
                    if d.page_content not in existing_contents:
                        current_docs.append(d)
                        existing_contents.add(d.page_content)
            except Exception as e:
                logger.warning("Self-RAG 精炼检索失败 (%s)，终止循环", type(e).__name__)
                break

    # 最后一轮或精炼失败：返回当前答案（即使忠实度未达标）
    return {
        "answer": answer,
        "docs": current_docs,
        "rounds": len(scores),
        "faithfulness_scores": scores,
        "degraded": degraded,
    }
