"""Orca agent — pre-computation (generate .inp) and post-computation (parse NEVPT2 energy).

Same conditional routing pattern:
- artifacts_in empty → pre: read orbitals, generate .inp, generate slurm
- artifacts_in has log → post: parse NEVPT2 energy, register artifacts
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END

from contracts.orca_task import OrcaParams, OrcaResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class OrcaState(BaseAgentState[OrcaParams, OrcaResult]):
    pass


# ═══════════════════════════════════════════════════════════════════
# Pre-computation (TODO: domain developer implements)
# ═══════════════════════════════════════════════════════════════════

def read_orbitals(state: OrcaState) -> dict:
    """Read active orbital info (from PySCF) via ArtifactResolver."""
    state.scratchpad["orbitals_read"] = True
    return {"node_history": state.node_history + ["read_orbitals"]}


def generate_input(state: OrcaState) -> dict:
    """Generate Orca .inp from params + orbital info."""
    state.scratchpad["inp_generated"] = True
    return {"node_history": state.node_history + ["generate_input"]}


def generate_slurm(state: OrcaState) -> dict:
    """Generate rough Slurm for HPC refinement."""
    state.scratchpad["slurm_generated"] = True
    return {"node_history": state.node_history + ["generate_slurm"]}


def pre_done(state: OrcaState) -> dict:
    from contracts.agent_task import Artifact, TaskStatus
    state.artifacts_out = [
        Artifact(artifact_id=f"{state.task_id}-inp", type="inp",
                 description="Orca input file", producer_agent="orca",
                 producer_task_id=state.task_id),
        Artifact(artifact_id=f"{state.task_id}-slurm", type="slurm",
                 description="Rough Slurm template", producer_agent="orca",
                 producer_task_id=state.task_id),
    ]
    return {"status": TaskStatus.DONE, "node_history": state.node_history + ["pre_done"]}


# ═══════════════════════════════════════════════════════════════════
# Post-computation (TODO: domain developer implements)
# ═══════════════════════════════════════════════════════════════════

def read_output(state: OrcaState) -> dict:
    """Read Orca .log produced by HPC execution."""
    return {"node_history": state.node_history + ["read_output"]}


def parse_energy(state: OrcaState) -> dict:
    """Parse NEVPT2/CC energy from Orca output."""
    return {
        "result_data": OrcaResult(),
        "node_history": state.node_history + ["parse_energy"],
    }


def register_artifacts(state: OrcaState) -> dict:
    from contracts.agent_task import Artifact, TaskStatus
    state.artifacts_out = [
        Artifact(artifact_id=f"{state.task_id}-log", type="log",
                 description="Orca NEVPT2 output log",
                 producer_agent="orca", producer_task_id=state.task_id),
    ]
    return {"status": TaskStatus.DONE, "node_history": state.node_history + ["register_artifacts"]}


# ═══════════════════════════════════════════════════════════════════

def route_entry(state: OrcaState) -> Literal["pre", "post"]:
    if state.task is None:
        return "pre"
    for ref in state.task.artifacts_in:
        if ref.type in ("log",):
            return "post"
    return "pre"


def build_orca_graph() -> StateGraph:
    graph = StateGraph(OrcaState)

    graph.add_node("read_orbitals", read_orbitals)
    graph.add_node("generate_input", generate_input)
    graph.add_node("generate_slurm", generate_slurm)
    graph.add_node("pre_done", pre_done)
    graph.add_node("read_output", read_output)
    graph.add_node("parse_energy", parse_energy)
    graph.add_node("register_artifacts", register_artifacts)

    graph.add_conditional_edges(START, route_entry, {"pre": "read_orbitals", "post": "read_output"})

    graph.add_edge("read_orbitals", "generate_input")
    graph.add_edge("generate_input", "generate_slurm")
    graph.add_edge("generate_slurm", "pre_done")
    graph.add_edge("pre_done", END)

    graph.add_edge("read_output", "parse_energy")
    graph.add_edge("parse_energy", "register_artifacts")
    graph.add_edge("register_artifacts", END)

    return graph


orca_graph = build_orca_graph().compile()


def register() -> AgentRegistration:
    return AgentRegistration(
        name="orca",
        subgraph=orca_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=2),
        timeout_s=120,
        dependencies=set(),
        description=(
            "Orca NEVPT2/CC agent. Pre: read orbitals → generate .inp → slurm template. "
            "Post: parse NEVPT2 energy → register artifacts."
        ),
    )
