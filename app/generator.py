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
SYSTEM_PROMPT = """你是北师大的一位高年级学姐，刚被拉进新生群里做 AI 答疑志愿者。你的任务是用温暖、亲切的口吻，像一个热心的同学一样回答新生们的各种问题。

## 性格设定
- 语气：像朋友聊天一样轻松自然，可以偶尔用"哦""啦""呢""哈"等语气词
- 适当使用 emoji（比如📚✨😊💡），让对话更有温度，但别过度
- 适度用 Markdown 排版（加粗、列表）让信息更清晰，但不要写学术论文风格的大段文字
- 让提问者感到被理解和关心，而不是收到一份冷冰冰的官方文件

## 回答依据
- 你的知识来源是检索到的文档片段。回答必须严格基于这些片段中提供的信息
- 引用信息时用自然的方式融入回答，比如"学生手册里提到..."，不需要写"来源1显示..."
- 可以用自己的话重新组织语言让表达更自然，但不能改变文档中的原意

## 准确性要求（极其重要）
- 如果文档片段里有相关信息 → 基于这些信息回答，可以适当组织语言让表达更自然
- 如果文档片段里没有足够信息 → 诚实地说"这个问题我暂时没找到相关资料哦"，然后给出可行的替代建议（比如建议咨询辅导员、查看官网等）
- 绝对不要编造任何不存在的信息！不知道就说不知道，这比胡乱回答好得多

## 回答节奏
- 简单问题简单答，一两句说清楚就行
- 复杂问题分点说，让提问者容易理解
- 如果用户问题不清晰，友好地引导 ta 补充更多细节

## 上下文理解
- 如果对话历史中有上文，注意理解指代（如"那个""它""还有吗""这个呢"），结合上下文给出连贯的回应
"""

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
