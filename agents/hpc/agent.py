"""HPC agent — stub skeleton.

TODO: Domain developer implements:
- collect_needs: read all JobRequests, validate input files exist via ArtifactResolver
- query_resources: call hpc-mcp queue-status for node availability
- refine_slurm: merge rough Slurm + cluster info → final Slurm
- submit: sbatch via hpc-mcp, record job_ids

Cleanup before retry: scancel all jobs, remove stale Slurm scripts.
"""
from langgraph.graph import StateGraph, START, END

from contracts.hpc_task import HPCParams, HPCResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class HPCState(BaseAgentState[HPCParams, HPCResult]):
    """HPC agent internal state."""
    pass


def stub_node(state: HPCState) -> dict:
    state.scratchpad["stub"] = True
    return {
        "status": state.status.__class__.DONE,
        "result_data": HPCResult(),
        "node_history": state.node_history + ["stub"],
    }


hpc_graph = (
    StateGraph(HPCState)
    .add_node("stub", stub_node)
    .add_edge(START, "stub")
    .add_edge("stub", END)
    .compile()
)


def register() -> AgentRegistration:
    return AgentRegistration(
        name="hpc",
        subgraph=hpc_graph,
        retry_policy=RetryPolicy(
            is_idempotent=False,
            max_retries=2,
            requires_cleanup_before_retry=True,
            cleanup_steps=["scancel old jobs", "remove stale Slurm scripts"],
        ),
        timeout_s=60,  # squeue查询 + slurm精化 + sbatch，秒级完成
        dependencies=set(),  # no chemical dependencies — dispatched in parallel
        description="HPC infrastructure agent — Slurm generation, resource query, job submission",
    )
