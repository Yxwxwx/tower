"""PySCF agent — RHF/DFT and CASSCF with pre/post routing.

Pre-computation:
- n_active_electrons==0 → RHF/DFT: generate_rhf_input → generate_slurm → pre_done
- n_active_electrons>0  → CASSCF: read_fchk → select_orbitals → generate_casscf_input
                          → generate_slurm → pre_done

Post-computation (shared):
- read_output → parse_energy → register_artifacts

Routing:
- artifacts_in has log → post path
- params.job_type==CASSCF → CASSCF pre path
- else → RHF/DFT pre path
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END

from contracts.pyscf_task import PySCFParams, PySCFResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy
from agents.pyscf.nodes import (
    # RHF/DFT
    generate_rhf_input,
    # CASSCF
    read_fchk,
    select_orbitals,
    generate_casscf_input,
    # Shared
    generate_slurm,
    pre_done,
    read_output,
    parse_energy,
    register_artifacts,
)


class PySCFState(BaseAgentState[PySCFParams, PySCFResult]):
    pass


# ═══════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════

def route_entry(state: PySCFState) -> Literal["post", "pre_rhf", "pre_casscf"]:
    """Route to correct path based on context.

    - Has log artifact → post-computation
    - CASSCF params → pre CASSCF path
    - Otherwise → pre RHF/DFT path
    """
    if state.task is not None:
        for ref in state.task.artifacts_in:
            if ref.type in ("log",):
                return "post"

        params = state.task.params
        if params.job_type == "CASSCF" or params.n_active_electrons > 0:
            return "pre_casscf"

    return "pre_rhf"


# ═══════════════════════════════════════════════════════════════════

def _finalize(state: PySCFState) -> dict:
    return {"agent_result": state.to_agent_result("pyscf")}


def build_pyscf_graph() -> StateGraph:
    graph = StateGraph(PySCFState)

    # Pre nodes
    graph.add_node("generate_rhf_input", generate_rhf_input)
    graph.add_node("read_fchk", read_fchk)
    graph.add_node("select_orbitals", select_orbitals)
    graph.add_node("generate_casscf_input", generate_casscf_input)
    graph.add_node("generate_slurm", generate_slurm)
    graph.add_node("pre_done", pre_done)

    # Post nodes
    graph.add_node("read_output", read_output)
    graph.add_node("parse_energy", parse_energy)
    graph.add_node("register_artifacts", register_artifacts)

    # Finalize
    graph.add_node("finalize", _finalize)

    # Entry routing
    graph.add_conditional_edges(START, route_entry, {
        "post": "read_output",
        "pre_rhf": "generate_rhf_input",
        "pre_casscf": "read_fchk",
    })

    # RHF/DFT pre chain
    graph.add_edge("generate_rhf_input", "generate_slurm")
    # CASSCF pre chain
    graph.add_edge("read_fchk", "select_orbitals")
    graph.add_edge("select_orbitals", "generate_casscf_input")
    graph.add_edge("generate_casscf_input", "generate_slurm")
    # Both converge → slurm → done
    graph.add_edge("generate_slurm", "pre_done")
    graph.add_edge("pre_done", "finalize")

    # Post chain
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
            "PySCF agent. RHF/DFT: generate script → slurm. "
            "CASSCF: read fchk → orbitals → script → slurm. "
            "Post: parse energy → register artifacts."
        ),
    )
