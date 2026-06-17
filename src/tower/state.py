from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Tower agent global state.

    Design principle: state holds data needed for the current workflow.
    Long-lived data lives in the memory layer.
    """

    # ── Messages ──
    messages: Annotated[list, add_messages]

    # ── Task & Plan ──
    task: str
    plan: list[dict]
    # plan[i] = {
    #     "name": str,            # tool name
    #     "args": dict,           # tool parameters
    #     "id": str,              # LLM-generated tool_call_id
    #     "step_type": str,       # [NEW] "computation" | "analysis" | "io"
    #     "expected_output": str, # [NEW] human-readable expected result
    # }
    current_step_index: int

    # ── Tool Execution ──
    tool_results: dict[str, Any]
    tool_history: list[dict]       # [NEW] explicit invocation history

    # ── Observation & Error ──
    observation: str
    error_info: dict | None        # [NEW] structured error from detectors
    # error_info = {
    #     "error_type": str,       # "scf_not_converged" | "python_error" | ...
    #     "message": str,
    #     "suggestion": str,
    #     "can_auto_fix": bool,
    # }
    refinement_needed: bool
    retry_count: int
    max_retries: int
    retry_pending: bool            # refine triggered retry of current step

    # ── Background Task ──
    background_task: dict | None   # [NEW] track long-running calculations
    # background_task = {
    #     "task_id": str,
    #     "status": "running" | "completed" | "failed",
    #     "started_at": str,
    # }

    # ── Multi-Pass Control ──
    pass_count: int

    # ── Final ──
    final_response: str
    task_complete: bool
    node_history: list[str]

    # ── Runtime Injection (not persisted in checkpoint) ──
    _runtime: dict[str, Any]
