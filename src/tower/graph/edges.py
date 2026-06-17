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
        if plan:
            return "plan"
        else:
            return "respond"

    return "act"


def route_after_refine(state: AgentState) -> str:
    """refine 之后：如果可以自动修正且还有重试次数 → act，否则 → respond。"""
    error = state.get("error_info", {}) or {}
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if error.get("can_auto_fix") and retry_count < max_retries:
        return "act"
    if state.get("retry_pending"):
        return "act"
    return "respond"
