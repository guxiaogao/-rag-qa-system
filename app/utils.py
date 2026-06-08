"""
========== 工具函数模块 ==========
RAG 项目中各处复用的辅助函数。
"""

import re


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
