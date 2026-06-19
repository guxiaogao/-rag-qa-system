"""
========== 工具函数模块 ==========
RAG 项目中各处复用的辅助函数。
"""

import re
from functools import lru_cache

from langchain_openai import ChatOpenAI
from app.config import settings


# ========== LLM 实例缓存 ==========

@lru_cache(maxsize=8)
def get_cached_llm(model: str, temperature: float, *, _tag: str = "") -> ChatOpenAI:
    """
    创建（并缓存）ChatOpenAI 实例。
    使用 lru_cache 避免高并发下重复创建 HTTP 客户端。

    参数：
        model:       模型名称
        temperature: 生成温度（0=确定性）
        _tag:        可选标签，用于区分同模型同温度的不同用途（如 "judge" vs "rewrite"）
                     Python 的 lru_cache 以参数为 key，不传 _tag 时 (model, temp) 即可区分

    注意：
        - 同一 (model, temperature) 组合只会创建一次实例
        - 实例不含状态，线程安全
        - maxsize=8 覆盖：chat×3 温度 + judge×1 + rewrite×1 + refiner×1
    """
    return ChatOpenAI(
        model=model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=temperature,
        request_timeout=30.0,  # 防挂起：30s 超时
    )


# ========== 分数提取 ==========

def extract_score(text: str) -> float:
    """
    从 LLM 返回的文本中提取分数，统一归一化到 0-1。

    支持多种格式：
    - "8"           → 0.8（裸数字，>1 视为 10 分制）
    - "8/10"        → 0.8（明确的分母）
    - "0.85"        → 0.85（0-1 小数值）
    - "分数：7"      → 0.7（带中文前缀）
    - "Score: 8"    → 0.8（带英文前缀）

    无法解析时返回 0.5（保守的中性分）。
    """
    text = text.strip()

    # 尝试匹配 "X/10" 或 "X/5" 格式
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", text)
    if match:
        return float(match.group(1)) / float(match.group(2))

    # 尝试匹配 "分数：X" 或 "Score: X" 格式（带文字前缀，假设满分 10）
    match = re.search(
        r"(?:分数|得分|score|rating)[：:\s]*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        score = float(match.group(1))
        return score / 10.0 if score > 1 else score

    # 尝试匹配 0-1 的小数（如 "0.85"）
    match = re.search(r"\b(0\.\d+)\b", text)
    if match:
        return float(match.group(1))

    # 尝试匹配裸数字（模型可能直接返回数字，如 "10" / "8" / "7.5"）
    match = re.search(r"^\s*(\d+(?:\.\d+)?)\s*$", text)
    if match:
        score = float(match.group(1))
        # 大于 1 的视为 1-10 分制，归一化到 0-1
        return score / 10.0 if score > 1 else score

    # 默认 0.5（无法解析时给中等分）
    return 0.5
