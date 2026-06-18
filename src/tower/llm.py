"""共享 LLM 工厂 —— 所有 agent 通过此模块获取模型实例。

LangGraph 官方模式：
- 模块级 load_dotenv() 确保环境变量在 import 时加载
- get_model() 返回 ChatDeepSeek 实例
- bind_tools() 用于需要 MCP 工具调用的 agent
"""

import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 确保 .env 在任何 import 之前加载
load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# 默认模型工厂
# ═══════════════════════════════════════════════════════════════════


def get_model(
    model_name: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatDeepSeek:
    """获取 DeepSeek Chat 模型实例（LangGraph agent 节点中的默认用法）。

    Args:
        model_name: "deepseek-chat" (通用) 或 "deepseek-reasoner" (推理)
        temperature: 0.0 = 确定性输出，适用于结构化任务
        max_tokens: 最大输出 token 数

    Returns:
        ChatDeepSeek 实例，可直接用于 model.invoke(messages)
    """
    return ChatDeepSeek(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


# ═══════════════════════════════════════════════════════════════════
# 带工具绑定的模型
# ═══════════════════════════════════════════════════════════════════


def get_model_with_tools(
    tools: list,
    model_name: str = "deepseek-chat",
    temperature: float = 0.0,
) -> ChatDeepSeek:
    """获取绑定了工具的模型实例（用于需要 MCP 工具调用的 agent）。

    LangGraph 官方模式：model.bind_tools(tools) → agent 节点中 model.invoke(messages)
    工具调用通过 AIMessage.tool_calls 返回。

    Args:
        tools: LangChain Tool 对象列表
        model_name: "deepseek-chat" 或 "deepseek-reasoner"
        temperature: 模型温度

    Returns:
        绑定了工具的 ChatDeepSeek 实例
    """
    model = get_model(model_name=model_name, temperature=temperature)
    return model.bind_tools(tools)
