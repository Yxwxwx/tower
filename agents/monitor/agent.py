"""Monitor agent — queue polling, log parsing, error classification.

Event-driven, independent of main handoff chain.
Writes MonitorEvents to RunStateStore.event_log (append-only).
Supervisor reads events and decides retry/escalate.

Mock: assumes all jobs complete successfully on first poll.
"""
from langgraph.graph import StateGraph, START, END

from contracts.monitor_task import MonitorParams, MonitorResult, MonitorEvent
from contracts.agent_task import Artifact, ArtifactStatus, TaskStatus
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class MonitorState(BaseAgentState[MonitorParams, MonitorResult]):
    pass


# ═══════════════════════════════════════════════════════════════════

def poll_jobs(state: MonitorState) -> dict:
    """Query hpc-mcp queue-status for each watched job.

    Mock: all jobs instantly DONE.

    TODO: infra engineer wires real squeue/sacct polling.
    """
    watchlist = state.task.params.watchlist if state.task else {}
    events = []

    for job_id, agent_name in watchlist.items():
        # Mock: job completed successfully
        events.append(MonitorEvent(
            job_id=job_id,
            agent=agent_name,
            event_type="JOB_DONE",
            log_snippet="",
        ).model_dump())

    return {
        "node_history": state.node_history + ["poll_jobs"],
        "scratchpad": {**state.scratchpad, "events": events, "watchlist": watchlist},
    }


def parse_logs(state: MonitorState) -> dict:
    """Fetch and parse logs for completed jobs via hpc-mcp log-parser.

    Mock: all jobs succeeded, no logs needed.

    TODO: infra engineer wires real log fetching + error pattern matching.
    """
    events = state.scratchpad.get("events", [])
    log_ids = []

    for evt_dict in events:
        job_id = evt_dict.get("job_id", "")
        agent = evt_dict.get("agent", "")
        # In production: fetch log via hpc-mcp log-parser.fetch
        # For mock: create a log artifact ID
        log_id = f"log-{agent}-{job_id}"
        log_ids.append(log_id)

    return {
        "node_history": state.node_history + ["parse_logs"],
        "scratchpad": {**state.scratchpad, "log_ids": log_ids},
    }


def classify_errors(state: MonitorState) -> dict:
    """Classify any errors found in logs.

    Mock: no errors. In production: call log-parser.classify.

    TODO: infra engineer wires error classification (regex/LLM).
    """
    return {
        "node_history": state.node_history + ["classify_errors"],
    }


def post_done(state: MonitorState) -> dict:
    """Return log artifacts so supervisor can pass them back to domain agent."""
    log_ids = state.scratchpad.get("log_ids", [])
    events = [MonitorEvent(**e) for e in state.scratchpad.get("events", [])]

    return {
        "status": TaskStatus.DONE,
        "result_data": MonitorResult(
            events=events,
            completed_jobs=[e.get("job_id", "") for e in state.scratchpad.get("events", [])],
            summary="All jobs completed",
        ),
        "artifacts_out": [
            Artifact(artifact_id=log_id, path="", type="log",
                     description=f"Computation log",
                     producer_agent="monitor", producer_task_id=state.task_id,
                     status=ArtifactStatus.READY)
            for log_id in log_ids
        ],
        "node_history": state.node_history + ["post_done"],
    }


# ═══════════════════════════════════════════════════════════════════

def _finalize(state: MonitorState) -> dict:
    return {"agent_result": state.to_agent_result("monitor")}


monitor_graph = (
    StateGraph(MonitorState)
    .add_node("poll_jobs", poll_jobs)
    .add_node("parse_logs", parse_logs)
    .add_node("classify_errors", classify_errors)
    .add_node("post_done", post_done)
    .add_node("finalize", _finalize)
    .add_edge(START, "poll_jobs")
    .add_edge("poll_jobs", "parse_logs")
    .add_edge("parse_logs", "classify_errors")
    .add_edge("classify_errors", "post_done")
    .add_edge("post_done", "finalize")
    .add_edge("finalize", END)
    .compile()
)


def register() -> AgentRegistration:
    return AgentRegistration(
        name="monitor",
        subgraph=monitor_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=2),
        timeout_s=86400,  # long-running daemon
        dependencies=set(),
        description="Monitor agent — poll sacct, parse logs, classify errors, feedback to supervisor",
    )
