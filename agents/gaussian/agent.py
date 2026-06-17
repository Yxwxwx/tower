"""Gaussian agent — compiled LangGraph subgraph with conditional routing.

Two independent paths, routed by task context:
- artifacts_in is empty → pre-computation: query DB, generate input, generate slurm
- artifacts_in has log/fchk → post-computation: read output, parse, register artifacts

Supervisor calls the same agent twice in a single run:
1. First call: "generate input for N2 HF opt" → .gjf + rough slurm
2. HPC agent submits → Monitor watches → computation completes
3. Second call: "parse output of job 12345" → energy, convergence, fchk artifact
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END

from contracts.gaussian_task import GaussianParams, GaussianResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy
from agents.gaussian.nodes import (
    # pre-computation
    query_knowledge,
    generate_input,
    generate_slurm_template,
    pre_compute_done,
    # post-computation
    read_output,
    parse_energy,
    register_artifacts,
)


# ═══════════════════════════════════════════════════════════════════
# Agent state
# ═══════════════════════════════════════════════════════════════════

class GaussianState(BaseAgentState[GaussianParams, GaussianResult]):
    """Gaussian agent internal state — private, not visible to other agents."""
    pass


# ═══════════════════════════════════════════════════════════════════
# Routing: pre or post computation?
# ═══════════════════════════════════════════════════════════════════

def route_entry(state: GaussianState) -> Literal["pre", "post"]:
    """Decide which path to take based on artifacts_in.

    - No artifacts_in → supervisor wants us to generate input (pre-computation).
    - Has log/fchk in artifacts_in → supervisor wants us to parse output (post-computation).
    """
    task = state.task
    if task is None:
        return "pre"

    # Check if supervisor passed us output artifacts to parse
    for ref in task.artifacts_in:
        if ref.type in ("log", "fchk"):
            return "post"

    return "pre"


# ═══════════════════════════════════════════════════════════════════
# Finalize — converts internal state → AgentResult protocol
# ═══════════════════════════════════════════════════════════════════

def _finalize(state: GaussianState) -> dict:
    """Convert internal state to AgentResult protocol.

    This is the contract boundary. Every agent MUST return AgentResult.
    """
    result = state.to_agent_result("gaussian")
    return {"agent_result": result}


# ═══════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════

def build_gaussian_graph() -> StateGraph:
    graph = StateGraph(GaussianState)

    # Pre-computation path
    graph.add_node("query_knowledge", query_knowledge)
    graph.add_node("generate_input", generate_input)
    graph.add_node("generate_slurm_template", generate_slurm_template)
    graph.add_node("pre_compute_done", pre_compute_done)

    # Post-computation path
    graph.add_node("read_output", read_output)
    graph.add_node("parse_energy", parse_energy)
    graph.add_node("register_artifacts", register_artifacts)

    # Finalize — converts internal state to AgentResult protocol
    graph.add_node("finalize", _finalize)

    # Conditional entry: pre or post?
    graph.add_conditional_edges(START, route_entry, {
        "pre": "query_knowledge",
        "post": "read_output",
    })

    # Pre chain → finalize
    graph.add_edge("query_knowledge", "generate_input")
    graph.add_edge("generate_input", "generate_slurm_template")
    graph.add_edge("generate_slurm_template", "pre_compute_done")
    graph.add_edge("pre_compute_done", "finalize")

    # Post chain → finalize
    graph.add_edge("read_output", "parse_energy")
    graph.add_edge("parse_energy", "register_artifacts")
    graph.add_edge("register_artifacts", "finalize")

    graph.add_edge("finalize", END)

    return graph


gaussian_graph = build_gaussian_graph().compile()


# ═══════════════════════════════════════════════════════════════════

def register() -> AgentRegistration:
    return AgentRegistration(
        name="gaussian",
        subgraph=gaussian_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=2),
        timeout_s=120,  # 生成输入 + 粗糙slurm，或解析输出，都是秒级
        dependencies=set(),
        description=(
            "Gaussian computational chemistry agent. "
            "Pre-computation: query DB → generate input → slurm template. "
            "Post-computation: read log → parse energy → register artifacts."
        ),
    )
