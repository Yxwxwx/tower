from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from tower.state import AgentState
from tower.graph.nodes import plan_node, act_node, observe_node, refine_node, respond_node
from tower.graph.edges import route_after_plan, route_after_observe, route_after_refine


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """构建 Plan→Act→Observe→(Refine|Plan|Respond) 状态图。

    observe 之后如果当前轮步骤全部完成，回到 plan 让 LLM
    根据工具结果判断是否需要更多操作（多轮循环）。

    Args:
        checkpointer: LangGraph checkpointer（PostgresSaver 等）。
    """
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)
    graph.add_node("refine", refine_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"act": "act", "respond": "respond"},
    )
    graph.add_edge("act", "observe")

    graph.add_conditional_edges(
        "observe",
        route_after_observe,
        {
            "refine": "refine",
            "act": "act",
            "plan": "plan",  # 回到 plan 继续规划
            "respond": "respond",
        },
    )

    graph.add_conditional_edges(
        "refine",
        route_after_refine,
        {"act": "act", "respond": "respond"},
    )
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer)
