"""PySCF agent — pre-computation (generate script) and post-computation (parse output).

Same conditional routing pattern as gaussian agent:
- artifacts_in empty → pre: read fchk, select orbitals, generate CASSCF script
- artifacts_in has log → post: parse CASSCF energy, register orbital info artifact
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END

from contracts.pyscf_task import PySCFParams, PySCFResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class PySCFState(BaseAgentState[PySCFParams, PySCFResult]):
    pass


# ═══════════════════════════════════════════════════════════════════
# Pre-computation nodes (TODO: domain developer implements)
# ═══════════════════════════════════════════════════════════════════

def read_fchk(state: PySCFState) -> dict:
    """Read Gaussian checkpoint via ArtifactResolver. Extract MO coefficients."""
    state.scratchpad["fchk_read"] = True
    return {"node_history": state.node_history + ["read_fchk"]}


def select_orbitals(state: PySCFState) -> dict:
    """Select active space orbitals. Write active_orbitals.json."""
    state.scratchpad["orbitals_selected"] = True
    return {"node_history": state.node_history + ["select_orbitals"]}


def generate_slurm(state: PySCFState) -> dict:
    """Generate rough Slurm template for HPC refinement."""
    state.scratchpad["slurm_generated"] = True
    return {"node_history": state.node_history + ["generate_slurm"]}


def pre_done(state: PySCFState) -> dict:
    from contracts.agent_task import Artifact, TaskStatus
    state.artifacts_out = [
        Artifact(artifact_id=f"{state.task_id}-orbitals", type="json",
                 description="Active orbital selection", producer_agent="pyscf",
                 producer_task_id=state.task_id),
        Artifact(artifact_id=f"{state.task_id}-slurm", type="slurm",
                 description="Rough Slurm template", producer_agent="pyscf",
                 producer_task_id=state.task_id),
    ]
    return {"status": TaskStatus.DONE, "node_history": state.node_history + ["pre_done"]}


# ═══════════════════════════════════════════════════════════════════
# Post-computation nodes (TODO: domain developer implements)
# ═══════════════════════════════════════════════════════════════════

def read_output(state: PySCFState) -> dict:
    """Read CASSCF output log and orbital JSON."""
    return {"node_history": state.node_history + ["read_output"]}


def parse_energy(state: PySCFState) -> dict:
    """Parse CASSCF energy and CI coefficients from log."""
    return {
        "result_data": PySCFResult(),
        "node_history": state.node_history + ["parse_energy"],
    }


def register_artifacts(state: PySCFState) -> dict:
    """Register CASSCF output as artifacts for downstream (orca agent)."""
    from contracts.agent_task import Artifact, TaskStatus
    state.artifacts_out = [
        Artifact(artifact_id=f"{state.task_id}-casscf", type="json",
                 description="CASSCF energy + orbital info",
                 producer_agent="pyscf", producer_task_id=state.task_id),
    ]
    return {"status": TaskStatus.DONE, "node_history": state.node_history + ["register_artifacts"]}


# ═══════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════

def route_entry(state: PySCFState) -> Literal["pre", "post"]:
    if state.task is None:
        return "pre"
    for ref in state.task.artifacts_in:
        if ref.type in ("log", "json"):
            return "post"
    return "pre"


def _finalize(state: PySCFState) -> dict:
    result = state.to_agent_result("pyscf")
    return {"agent_result": result}


# ═══════════════════════════════════════════════════════════════════

def build_pyscf_graph() -> StateGraph:
    graph = StateGraph(PySCFState)

    graph.add_node("read_fchk", read_fchk)
    graph.add_node("select_orbitals", select_orbitals)
    graph.add_node("generate_slurm", generate_slurm)
    graph.add_node("pre_done", pre_done)
    graph.add_node("read_output", read_output)
    graph.add_node("parse_energy", parse_energy)
    graph.add_node("register_artifacts", register_artifacts)
    graph.add_node("finalize", _finalize)

    graph.add_conditional_edges(START, route_entry, {"pre": "read_fchk", "post": "read_output"})

    graph.add_edge("read_fchk", "select_orbitals")
    graph.add_edge("select_orbitals", "generate_slurm")
    graph.add_edge("generate_slurm", "pre_done")
    graph.add_edge("pre_done", "finalize")

    graph.add_edge("read_output", "parse_energy")
    graph.add_edge("parse_energy", "register_artifacts")
    graph.add_edge("register_artifacts", "finalize")

    graph.add_edge("finalize", END)

    return graph


pyscf_graph = build_pyscf_graph().compile()


def register() -> AgentRegistration:
    return AgentRegistration(
        name="pyscf",
        subgraph=pyscf_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=2),
        timeout_s=120,
        dependencies=set(),
        description=(
            "PySCF CASSCF agent. Pre: read fchk → select orbitals → generate script. "
            "Post: parse CASSCF energy → register orbital info artifact."
        ),
    )
