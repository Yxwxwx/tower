"""PySCF agent — stub skeleton.

TODO: Domain developer implements:
- read_fchk: read Gaussian checkpoint via ArtifactResolver
- select_orbitals: select active space, write active_orbitals.json
- run_casscf: execute CASSCF, write casscf_energy.json
- generate_slurm_template: rough Slurm for HPC
"""
from langgraph.graph import StateGraph, START, END

from contracts.pyscf_task import PySCFParams, PySCFResult
from tower_agent_kit.base import BaseAgentState, AgentRegistration, RetryPolicy


class PySCFState(BaseAgentState[PySCFParams, PySCFResult]):
    """PySCF agent internal state."""
    fchk_read: bool = False
    orbitals_selected: bool = False
    casscf_done: bool = False


def stub_node(state: PySCFState) -> dict:
    """Stub — domain developer replaces with real implementation."""
    state.scratchpad["stub"] = True
    return {
        "status": state.status.__class__.DONE,
        "result_data": PySCFResult(),
        "node_history": state.node_history + ["stub"],
    }


pyscf_graph = (
    StateGraph(PySCFState)
    .add_node("stub", stub_node)
    .add_edge(START, "stub")
    .add_edge("stub", END)
    .compile()
)


def register() -> AgentRegistration:
    return AgentRegistration(
        name="pyscf",
        subgraph=pyscf_graph,
        retry_policy=RetryPolicy(is_idempotent=True, max_retries=2),
        timeout_s=3600,
        dependencies=set(),
        description="PySCF CASSCF agent — orbital selection and multi-reference computation",
    )
