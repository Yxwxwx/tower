"""Gaussian agent — compiled LangGraph subgraph.

Implements the agent contract from docs/superpowers/contracts/agent_contract.md:
- Input: AgentTask[GaussianParams]
- Output: AgentResult[GaussianResult]
- Exposes: register() → AgentRegistration
"""
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from contracts.gaussian_task import GaussianParams, GaussianResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy
from agents.gaussian.nodes import (
    validate_params,
    write_input,
    generate_slurm_template,
    execute,
    parse_output,
)


# ═══════════════════════════════════════════════════════════════════
# Gaussian agent state
# ═══════════════════════════════════════════════════════════════════

class GaussianState(BaseAgentState[GaussianParams, GaussianResult]):
    """Gaussian agent internal state.

    Private fields are NOT visible to other agents.
    Only AgentResult[GaussianResult] is exposed externally.
    """
    # ── Execution flags ──
    input_written: bool = False
    slurm_generated: bool = False
    executed: bool = False
    output_parsed: bool = False

    # ── Paths (private to this agent) ──
    gjf_path: str = ""
    slurm_path: str = ""


# ═══════════════════════════════════════════════════════════════════
# Graph compilation
# ═══════════════════════════════════════════════════════════════════

def build_gaussian_graph() -> StateGraph:
    """Build the Gaussian agent subgraph.

    Topology:
        START → validate_params → write_input → generate_slurm_template
              → execute → parse_output → END

    On any failure: node sets status=FAILED, errors populated.
    Supervisor reads AgentResult and decides retry/escalate.
    """
    graph = StateGraph(GaussianState)

    graph.add_node("validate_params", validate_params)
    graph.add_node("write_input", write_input)
    graph.add_node("generate_slurm_template", generate_slurm_template)
    graph.add_node("execute", execute)
    graph.add_node("parse_output", parse_output)

    # Linear pipeline: each node depends on the previous
    graph.add_edge(START, "validate_params")
    graph.add_edge("validate_params", "write_input")
    graph.add_edge("write_input", "generate_slurm_template")
    graph.add_edge("generate_slurm_template", "execute")
    graph.add_edge("execute", "parse_output")
    graph.add_edge("parse_output", END)

    return graph


gaussian_graph = build_gaussian_graph().compile()


# ═══════════════════════════════════════════════════════════════════
# Registration — implements agent contract
# ═══════════════════════════════════════════════════════════════════

def register() -> AgentRegistration:
    """Expose agent metadata for the framework.

    Called by supervisor or agent loader to discover this agent.
    """
    return AgentRegistration(
        name="gaussian",
        subgraph=gaussian_graph,
        retry_policy=RetryPolicy(
            is_idempotent=True,
            max_retries=2,
            escalate_after_max=True,
        ),
        timeout_s=120,  # 生成输入文件 + 粗糙slurm模板，秒级完成
        dependencies=set(),  # no upstream dependencies
        description=(
            "Gaussian computational chemistry agent. "
            "Generates input files, runs HF/DFT calculations, "
            "outputs converged wavefunction checkpoints."
        ),
    )
