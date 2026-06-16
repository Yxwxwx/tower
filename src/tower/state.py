from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Tower agent 的全局 state。

    设计原则：state 只放当前工作流需要的数据。
    长期记忆走 memory 层，不放在 state 里。
    """

    messages: Annotated[list, add_messages]
    task: str
    plan: list[dict]  # list of {"name": "bash", "args": {...}, "id": "..."}
    current_step_index: int
    tool_results: dict[str, Any]
    observation: str
    refinement_needed: bool
    retry_pending: bool
    retry_count: int
    max_retries: int
    pass_count: int  # 多轮循环计数，plan→act→observe→plan 算一轮
    final_response: str
    task_complete: bool
    node_history: list[str]
