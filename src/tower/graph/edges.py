from tower.state import AgentState


def route_after_plan(state: AgentState) -> str:
    """plan 之后：task_complete → respond，有步骤 → act，空 plan → respond。"""
    if state.get("task_complete"):
        return "respond"
    plan = state.get("plan", [])
    if not plan:
        return "respond"
    return "act"


def route_after_observe(state: AgentState) -> str:
    """observe 之后：refine（重试） / act（下一步） / plan（继续规划） / respond。"""
    if state.get("refinement_needed"):
        return "refine"

    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])

    if idx >= len(plan):
        # 当前 plan 的步骤全部执行完
        if plan:
            # 有 plan → 回 plan 判断是否还需要更多工具
            return "plan"
        else:
            # plan 为空（闲聊或单步任务）→ 直接 respond，省一次 LLM 调用
            return "respond"

    return "act"
