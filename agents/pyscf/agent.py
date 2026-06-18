"""PySCF agent — LLM-driven script generation, parsing, and error correction.

Topology:
  pre:  generate_input → generate_slurm → pre_done
  post: read_output → parse_output → [register_artifacts | fix_input → pre_done]
        fix_input loops back via pre_done (supervisor re-dispatches hpc → monitor → post)
        max 5 correction attempts, then NEEDS_HUMAN.

Routing:
  - artifacts_in has log → post path
  - otherwise → pre path
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END

from contracts.pyscf_task import PySCFParams, PySCFResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy
from agents.pyscf.nodes import (
    generate_input,
    generate_slurm,
    pre_done,
    read_output,
    parse_output,
    register_artifacts,
    fix_input,
    should_retry,
)


class PySCFState(BaseAgentState[PySCFParams, PySCFResult]):
    pass


# ═══════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════


def route_entry(state: PySCFState) -> Literal["post", "pre"]:
    """Route to pre-computation or post-computation based on artifacts_in."""
    if state.task is not None:
        for ref in state.task.artifacts_in:
            if ref.type in ("log",):
                return "post"
    return "pre"


# ═══════════════════════════════════════════════════════════════════


def _finalize(state: PySCFState) -> dict:
    return {"agent_result": state.to_agent_result("pyscf")}


# ═══════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════


def build_pyscf_graph() -> StateGraph:
    graph = StateGraph(PySCFState)

    # Pre nodes
    graph.add_node("generate_input", generate_input)
    graph.add_node("generate_slurm", generate_slurm)
    graph.add_node("pre_done", pre_done)

    # Post nodes
    graph.add_node("read_output", read_output)
    graph.add_node("parse_output", parse_output)
    graph.add_node("register_artifacts", register_artifacts)
    graph.add_node("fix_input", fix_input)

    # Finalize
    graph.add_node("finalize", _finalize)

    # Entry routing
    graph.add_conditional_edges(START, route_entry, {
        "post": "read_output",
        "pre": "generate_input",
    })

    # Pre chain: generate_input → generate_slurm → pre_done → finalize
    graph.add_edge("generate_input", "generate_slurm")
    graph.add_edge("generate_slurm", "pre_done")
    graph.add_edge("pre_done", "finalize")

    # Post chain: read → parse → [register | fix]
    graph.add_edge("read_output", "parse_output")
    graph.add_conditional_edges("parse_output", should_retry, {
        "register": "register_artifacts",
        "fix": "fix_input",
    })
    graph.add_edge("register_artifacts", "finalize")
    # fix_input → pre_done → finalize (supervisor will re-dispatch hpc → monitor → post)
    graph.add_edge("fix_input", "pre_done")

    graph.add_edge("finalize", END)

    return graph


pyscf_graph = build_pyscf_graph().compile()


# ═══════════════════════════════════════════════════════════════════


def register() -> AgentRegistration:
    return AgentRegistration(
        name="pyscf",
        subgraph=pyscf_graph,
        retry_policy=RetryPolicy(
            is_idempotent=True,
            max_retries=5,
            requires_cleanup_before_retry=False,
        ),
        timeout_s=120,
        dependencies=set(),
        description=(
            "PySCF agent. Pre: LLM generates complete Python script "
            "(any method: RHF/UHF/DFT/CASSCF/MP2/CC/TD-DFT/opt/freq). "
            "Post: LLM parses output exhaustively. "
            "Auto-corrects errors up to 5 times."
        ),
    )
