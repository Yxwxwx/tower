"""Orca agent — stub skeleton.

TODO: Domain developer implements:
- read_orbitals: read active_orbitals.json via ArtifactResolver
- write_input: render Orca .inp from params + orbital info
- generate_slurm_template: rough Slurm for HPC
- run_or_submit: execute or delegate to HPC
"""
from langgraph.graph import StateGraph, START, END

from contracts.orca_task import OrcaParams, OrcaResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class OrcaState(BaseAgentState[OrcaParams, OrcaResult]):
    """Orca agent internal state."""
    pass


def stub_node(state: OrcaState) -> dict:
    state.scratchpad["stub"] = True
    return {
        "status": state.status.__class__.DONE,
        "result_data": OrcaResult(),
        "node_history": state.node_history + ["stub"],
    }


orca_graph = (
    StateGraph(OrcaState)
    .add_node("stub", stub_node)
    .add_edge(START, "stub")
    .add_edge("stub", END)
    .compile()
)


def register() -> AgentRegistration:
    return AgentRegistration(
        name="orca",
        subgraph=orca_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=2),
        timeout_s=7200,
        dependencies=set(),
        description="Orca NEVPT2/CC agent — post-HF multi-reference computation",
    )
