"""Monitor agent — stub skeleton.

TODO: Domain developer implements:
- poll_jobs: query hpc-mcp queue-status for each job in watchlist
- parse_logs: if job FAILED, fetch log via hpc-mcp log-parser
- classify_error: parse log → error_category + suggestion
- append_events: write MonitorEvent to RunStateStore.event_log (append-only)

Monitor writes events; it NEVER modifies agent state directly.
Supervisor reads events and decides retry/escalate.
"""
from langgraph.graph import StateGraph, START, END

from contracts.monitor_task import MonitorParams, MonitorResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class MonitorState(BaseAgentState[MonitorParams, MonitorResult]):
    """Monitor agent internal state."""
    pass


def stub_node(state: MonitorState) -> dict:
    state.scratchpad["stub"] = True
    return {
        "status": state.status.__class__.DONE,
        "result_data": MonitorResult(),
        "node_history": state.node_history + ["stub"],
    }


monitor_graph = (
    StateGraph(MonitorState)
    .add_node("stub", stub_node)
    .add_edge(START, "stub")
    .add_edge("stub", END)
    .compile()
)


def register() -> AgentRegistration:
    return AgentRegistration(
        name="monitor",
        subgraph=monitor_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=2),
        timeout_s=86400,  # long-running daemon
        dependencies=set(),  # independent, event-driven
        description="Monitor infrastructure agent — queue polling, log parsing, error feedback",
    )
