"""HPC agent — real cluster submission with Slurm, fallback to local bash.

Topology:
  analyze_artifacts → generate_slurm → submit → pre_done

All analysis and generation is LLM-powered. Real sbatch or local bash fallback.
"""
from langgraph.graph import StateGraph, START, END

from contracts.hpc_task import HPCParams, HPCResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy
from agents.hpc.nodes import (
    analyze_artifacts,
    generate_slurm,
    submit,
    pre_done,
)


class HPCState(BaseAgentState[HPCParams, HPCResult]):
    pass


def _finalize(state: HPCState) -> dict:
    return {"agent_result": state.to_agent_result("hpc")}


hpc_graph = (
    StateGraph(HPCState)
    .add_node("analyze_artifacts", analyze_artifacts)
    .add_node("generate_slurm", generate_slurm)
    .add_node("submit", submit)
    .add_node("pre_done", pre_done)
    .add_node("finalize", _finalize)
    .add_edge(START, "analyze_artifacts")
    .add_edge("analyze_artifacts", "generate_slurm")
    .add_edge("generate_slurm", "submit")
    .add_edge("submit", "pre_done")
    .add_edge("pre_done", "finalize")
    .add_edge("finalize", END)
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
            cleanup_steps=["scancel old jobs"],
        ),
        timeout_s=120,
        dependencies=set(),
        description=(
            "HPC agent — analyze cluster resources, generate Slurm scripts, "
            "submit via sbatch (or local bash fallback). Returns job ID and log artifact."
        ),
    )
