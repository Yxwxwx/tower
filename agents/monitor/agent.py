"""Monitor agent — real Slurm job polling via sacct/squeue.

Pipeline:
    poll_jobs → read_logs → post_done

Polls watched jobs until terminal, reads computation logs,
returns content as artifact for downstream domain agent.
"""
import subprocess
import time
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

from contracts.monitor_task import MonitorParams, MonitorResult, MonitorEvent
from contracts.agent_task import Artifact, ArtifactStatus, TaskStatus
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy

console = Console()


class MonitorState(BaseAgentState[MonitorParams, MonitorResult]):
    pass


def _update(state, **kwargs) -> dict:
    s = state.scratchpad if isinstance(state.scratchpad, dict) else {}
    return {"scratchpad": {**s, **kwargs}}


def _node_history(state, name: str) -> list:
    nh = []
    if hasattr(state, "node_history") and isinstance(state.node_history, list):
        nh = list(state.node_history)
    return nh + [name]


# ═══════════════════════════════════════════════════════════════════
# Slurm query helpers
# ═══════════════════════════════════════════════════════════════════


def _query_job(job_id: str) -> str:
    """Query Slurm job state via sacct or squeue. Returns state string."""
    try:
        r = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().split("\n"):
                if ".batch" not in line and line.strip():
                    return line.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "-o", "%T"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "DEADLINE",
            "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED"}
BAD = {"FAILED", "TIMEOUT", "CANCELLED", "DEADLINE",
       "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}


# ═══════════════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════════════


def poll_jobs(state: MonitorState) -> dict:
    """Poll Slurm jobs until all terminal or max wait exceeded."""
    if not hasattr(state, "task") or state.task is None:
        return {"node_history": _node_history(state, "poll_jobs")}

    params = state.task.params
    watchlist = getattr(params, "watchlist", {})
    if not watchlist:
        return {"node_history": _node_history(state, "poll_jobs")}

    poll_s = getattr(params, "poll_interval_s", 10) or 10
    max_wait = getattr(params, "max_wait_s", 86400) or 86400

    console.print()
    console.print(Panel(
        f"Watching {len(watchlist)} job(s): {', '.join(watchlist.keys())}\n"
        f"Poll interval: {poll_s}s · Max wait: {max_wait // 3600}h",
        title="[bold magenta]◉ Monitor[/]",
        border_style="magenta", padding=(0, 2),
    ))

    job_states = {}
    start = time.time()
    n_polls = 0

    for jid in watchlist:
        s = _query_job(jid)
        job_states[jid] = s
        console.print(f"  [dim]Job {jid}:[/] {s}")

    while True:
        if all(js in TERMINAL for js in job_states.values()):
            break
        if time.time() - start > max_wait:
            console.print(f"  [yellow]Max wait exceeded[/]")
            break
        time.sleep(poll_s)
        n_polls += 1
        for jid in list(watchlist.keys()):
            if job_states.get(jid, "") in TERMINAL:
                continue
            new_s = _query_job(jid)
            if new_s != job_states.get(jid, ""):
                job_states[jid] = new_s
                icon = "✕" if new_s in BAD else "✓" if new_s == "COMPLETED" else "◌"
                console.print(
                    f"  [{icon}] [dim]Job {jid}:[/] {new_s}  "
                    f"[dim](poll #{n_polls}, {time.time() - start:.0f}s)[/]"
                )

    # Build events (as plain dicts for safe serialization)
    events = []
    for jid, agent in watchlist.items():
        st = job_states.get(jid, "UNKNOWN")
        events.append({
            "job_id": jid, "agent": agent,
            "event_type": "JOB_FAILED" if st in BAD else
                          "JOB_DONE" if st == "COMPLETED" else "JOB_STARTED",
            "error_category": "timeout" if st == "TIMEOUT" else "unknown" if st in BAD else "",
            "suggestion": f"See slurm-{jid}.out for details" if st in BAD else "",
        })

    n_done = sum(1 for e in events if e["event_type"] == "JOB_DONE")
    n_fail = sum(1 for e in events if e["event_type"] == "JOB_FAILED")
    console.print(
        f"\n  [bold]{n_done} done, {n_fail} failed[/] · "
        f"{n_polls} polls · {time.time() - start:.0f}s elapsed"
    )

    return {
        "node_history": _node_history(state, "poll_jobs"),
        "scratchpad": {**(state.scratchpad if hasattr(state, "scratchpad") else {}),
                       "events": events, "job_states": job_states},
    }


def read_logs(state: MonitorState) -> dict:
    """Read computation log from job directory."""
    sp = state.scratchpad if hasattr(state, "scratchpad") and isinstance(state.scratchpad, dict) else {}
    params = state.task.params if state.task else MonitorParams()
    run_dir = getattr(params, "run_dir", "")
    log_path = getattr(params, "log_path", "pyscf.log") or "pyscf.log"

    log_content = ""
    if run_dir:
        f = Path(run_dir) / log_path
        if f.exists():
            log_content = f.read_text()
            console.print(f"  [dim]Read log: {f} ({len(log_content)} chars)[/]")
        else:
            console.print(f"  [yellow]Log not found: {f}[/]")

    return {
        "node_history": _node_history(state, "read_logs"),
        "scratchpad": {**sp, "log_content": log_content},
    }


def post_done(state: MonitorState) -> dict:
    """Return log artifact to supervisor."""
    sp = state.scratchpad if hasattr(state, "scratchpad") and isinstance(state.scratchpad, dict) else {}
    events_raw = sp.get("events", [])
    log_content = sp.get("log_content", "")
    params = state.task.params if state.task else MonitorParams()

    # Reconstruct typed events
    events = []
    for e in events_raw:
        try:
            events.append(MonitorEvent(**e))
        except Exception:
            pass

    completed = [e.job_id for e in events if e.event_type == "JOB_DONE"]
    failed = [e.job_id for e in events if e.event_type == "JOB_FAILED"]

    result = MonitorResult(
        events=events,
        completed_jobs=completed,
        failed_jobs=failed,
        log_content=log_content,
        summary=f"{len(completed)} done, {len(failed)} failed · {len(log_content)} chars log",
    )

    artifacts = []
    if log_content:
        artifacts.append(Artifact(
            artifact_id=f"{state.task_id}-log",
            path=str(Path(getattr(params, 'run_dir', '')) / getattr(params, 'log_path', 'pyscf.log')),
            type="log",
            description=log_content,
            producer_agent="monitor",
            producer_task_id=state.task_id,
            status=ArtifactStatus.READY,
        ))

    return {
        "status": TaskStatus.DONE,
        "result_data": result,
        "artifacts_out": artifacts,
        "node_history": _node_history(state, "post_done"),
    }


def _finalize(state: MonitorState) -> dict:
    return {"agent_result": state.to_agent_result("monitor")}


monitor_graph = (
    StateGraph(MonitorState)
    .add_node("poll_jobs", poll_jobs)
    .add_node("read_logs", read_logs)
    .add_node("post_done", post_done)
    .add_node("finalize", _finalize)
    .add_edge(START, "poll_jobs")
    .add_edge("poll_jobs", "read_logs")
    .add_edge("read_logs", "post_done")
    .add_edge("post_done", "finalize")
    .add_edge("finalize", END)
    .compile()
)


def register() -> AgentRegistration:
    return AgentRegistration(
        name="monitor",
        subgraph=monitor_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=1),
        timeout_s=86400,
        dependencies=set(),
        description=(
            "Monitor agent — polls Slurm jobs via sacct/squeue, "
            "reads computation logs, returns content for downstream parsing."
        ),
    )
