"""
========== 生成模块 ==========
负责将检索到的文档片段和用户问题一起交给 LLM，生成最终答案。
使用 DashScope（通义千问）的 OpenAI 兼容接口。
"""

from typing import List, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.exceptions import GenerationException
from app.utils import get_cached_llm


# ========== Prompt 模板 ==========

# 系统提示词：告诉 LLM 如何基于检索到的内容回答问题
SYSTEM_PROMPT = """你是北师大的AI答疑助手，用友好亲切的口吻回答新生问题。

要求：
1. 回答必须严格基于提供的文档片段，不编造信息
2. 找不到相关信息时诚实告知，并给出替代建议（咨询辅导员、查看官网等）
3. 简单问题简洁答，复杂问题分点说
4. 注意理解对话历史中的指代（"那个""它"等），给出连贯回应"""

# Chat prompt 模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", "{history}"),
    ("system", "{context}"),
    ("human", "{question}"),
])


def format_conversation_history(history: list[dict]) -> str:
    """
    将多轮对话历史格式化为 LLM 可读的文本。
    仅取最近 3 轮（6 条消息），避免 token 浪费。

    参数：
        history: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]

    返回：
        格式化的历史文本，无历史时返回空字符串。
    """
    if not history:
        return ""

    # 最多保留最近 3 轮对话 = 6 条消息
    recent = history[-6:]
    lines = ["[对话历史]"]
    for m in recent:
        label = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{label}：{m['content']}")
    return "\n".join(lines)


def get_llm(model: str = None, temperature: float = None) -> ChatOpenAI:
    """
    获取（缓存）DashScope 通义千问 LLM 实例。

    参数：
        model: 模型名称，默认用配置中的 chat_model
        temperature: 生成随机性（0-1），默认用配置中的 llm_temperature

    返回：
        缓存的 ChatOpenAI 实例。同一 (model, temperature) 组合全局复用。
    """
    if temperature is None:
        temperature = settings.llm_temperature
    return get_cached_llm(
        model=model or settings.chat_model,
        temperature=temperature,
    )


def format_context(docs: List[LCDocument]) -> str:
    """
    将检索到的文档片段格式化成 LLM 能看到的上下文文本。
    每个片段标注来源文件名。
    """
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("filename", "未知来源")
        parts.append(f"[来源 {i+1}] ({source}):\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _classify_generation_error(e: Exception, model: str = None) -> GenerationException:
    """
    将底层异常分类为友好的 GenerationException。
    根据异常消息中的关键字匹配常见错误模式。
    """
    error_str = str(e)
    if "api_key" in error_str.lower() or "authentication" in error_str.lower() or "401" in error_str:
        friendly_msg = "LLM API Key 无效或已过期，请检查 DASHSCOPE_API_KEY 配置"
    elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
        friendly_msg = "LLM 服务响应超时，请稍后重试"
    elif "rate" in error_str.lower() or "429" in error_str:
        friendly_msg = "LLM 服务请求过于频繁，请稍后重试"
    elif "model" in error_str.lower() or "404" in error_str:
        friendly_msg = f"LLM 模型不可用，请检查 CHAT_MODEL 配置（当前: {model or settings.chat_model}）"
    else:
        friendly_msg = "LLM 生成回答失败，请稍后重试"
    return GenerationException(
        message=friendly_msg,
        detail=f"{type(e).__name__}: {error_str}",
    )


def generate_answer(
    query: str,
    docs: List[LCDocument],
    model: str = None,
    temperature: float = None,
    conversation_history: list[dict] = None,
) -> str:
    """
    给定问题 + 检索到的文档，让 LLM 生成回答（同步，阻塞式）。

    参数：
        query: 用户问题
        docs: 检索到的文档片段列表
        model: 可选的模型覆盖
        temperature: LLM 生成温度，默认用全局配置
        conversation_history: 多轮对话历史，[{"role":"user","content":"..."}, ...]

    返回：
        LLM 生成的回答文本

    异常：
        GenerationException：LLM 调用失败（API Key 无效、超时、限流等）
    """
    try:
        llm = get_llm(model=model, temperature=temperature)
        context = format_context(docs)
        history = format_conversation_history(conversation_history or [])
        messages = chat_prompt.format_messages(
            history=history,
            context=f"以下是从知识库中检索到的 {len(docs)} 个相关片段：{context}",
            question=query,
        )
        response = llm.invoke(messages)
        return response.content
    except GenerationException:
        raise
    except Exception as e:
        raise _classify_generation_error(e, model)


async def generate_answer_stream(
    query: str,
    docs: List[LCDocument],
    model: str = None,
    temperature: float = None,
    conversation_history: list[dict] = None,
) -> AsyncGenerator[str, None]:
    """
    给定问题 + 检索到的文档，让 LLM 流式生成回答。

    与 generate_answer() 不同，此函数通过 llm.astream() 逐 token 产出，
    每个 token 是一个独立字符串，调用方可以用 async for 逐块接收。

    参数：
        query: 用户问题
        docs: 检索到的文档片段列表
        model: 可选的模型覆盖
        temperature: LLM 生成温度，默认用全局配置
        conversation_history: 多轮对话历史，[{"role":"user","content":"..."}, ...]

    Yields:
        str: 单个 token 文本（非空）

    异常：
        GenerationException：LLM 调用失败（API Key 无效、超时、限流等）
    """
    try:
        llm = get_llm(model=model, temperature=temperature)
        context = format_context(docs)
        history = format_conversation_history(conversation_history or [])
        messages = chat_prompt.format_messages(
            history=history,
            context=f"以下是从知识库中检索到的 {len(docs)} 个相关片段：{context}",
            question=query,
        )
        async for chunk in llm.astream(messages):
            # 过滤空 token（某些 chunk 的 content 可能为空字符串）
            if chunk.content:
                yield chunk.content
    except GenerationException:
        raise
    except Exception as e:
        raise _classify_generation_error(e, model)
