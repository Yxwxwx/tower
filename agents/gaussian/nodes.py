"""Gaussian agent nodes — each node is a pure function (state) → dict.

Nodes:
1. validate_params — validate GaussianParams before execution
2. write_input — generate .gjf input file from template
3. generate_slurm_template — create rough Slurm script for HPC refinement
4. execute — run Gaussian (local or hand off to HPC)
5. parse_output — parse .log for energy, convergence, errors
"""
import json
from datetime import datetime

from contracts.gaussian_task import GaussianParams, GaussianResult
from tower.state.artifact_registry import ArtifactRegistry


# ═══════════════════════════════════════════════════════════════════
# Node 1: Validate parameters
# ═══════════════════════════════════════════════════════════════════

def validate_params(state) -> dict:
    """Validate GaussianParams before proceeding.

    Checks enforced:
    - method is non-empty
    - basis is non-empty
    - memory_mb > 0
    - nprocs > 0

    TODO: domain developer implements full validation logic.
    """
    task = state.task
    if task is None:
        return {
            "status": state.status.__class__.FAILED,
            "errors": ["No AgentTask provided"],
        }

    params: GaussianParams = task.params
    errors = []

    if not params.method.strip():
        errors.append("method must be non-empty")
    if not params.basis.strip():
        errors.append("basis must be non-empty")
    if params.memory_mb <= 0:
        errors.append("memory_mb must be > 0")
    if params.nprocs <= 0:
        errors.append("nprocs must be > 0")

    if errors:
        return {"status": state.status.__class__.FAILED, "errors": errors}

    state.scratchpad["params_validated"] = True
    return {"node_history": state.node_history + ["validate_params"]}


# ═══════════════════════════════════════════════════════════════════
# Node 2: Write Gaussian input file
# ═══════════════════════════════════════════════════════════════════

def write_input(state) -> dict:
    """Generate .gjf input file from GaussianParams + template.

    Uses infra-mcp template.render tool to render the input file.
    Writes result via infra-mcp filesystem.write.

    TODO: domain developer implements actual MCP tool calls.
    """
    task = state.task
    params: GaussianParams = task.params

    # Render input from template (stub — domain dev implements)
    input_content = _render_gaussian_input(params, task.task_id)

    # Write to workspace (stub — domain dev implements MCP call)
    gjf_path = f"jobs/{task.task_id}/{task.task_id}.gjf"

    state.scratchpad["gjf_path"] = gjf_path
    state.scratchpad["gjf_content"] = input_content

    return {"node_history": state.node_history + ["write_input"]}


# ═══════════════════════════════════════════════════════════════════
# Node 3: Generate rough Slurm template
# ═══════════════════════════════════════════════════════════════════

def generate_slurm_template(state) -> dict:
    """Create a rough Slurm script for the HPC agent to refine.

    Includes #SBATCH directives for mem, cpus, walltime.
    HPC agent refines with partition, account, node selection.

    TODO: domain developer implements actual template rendering.
    """
    task = state.task
    params: GaussianParams = task.params

    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={task.task_id}
#SBATCH --mem={params.memory_mb}
#SBATCH --cpus-per-task={params.nprocs}
#SBATCH --time=24:00:00

# Rough template — HPC agent refines partition, account, node constraints

g16 < {state.scratchpad.get('gjf_path', 'input.gjf')} > jobs/{task.task_id}/{task.task_id}.log
"""

    slurm_path = f"jobs/{task.task_id}/{task.task_id}_rough.sh"
    state.scratchpad["slurm_path"] = slurm_path
    state.scratchpad["slurm_content"] = slurm_content

    return {"node_history": state.node_history + ["generate_slurm_template"]}


# ═══════════════════════════════════════════════════════════════════
# Node 4: Execute
# ═══════════════════════════════════════════════════════════════════

def execute(state) -> dict:
    """Execute Gaussian calculation.

    Two modes:
    a) Local execution: run g16 directly (for small test jobs)
    b) HPC handoff: produce artifacts and let HPC agent submit

    TODO: domain developer implements actual execution logic.
    """
    # For MVP: stub implementation — marks execution as complete
    # Domain developer replaces with actual g16 call or HPC handoff
    state.scratchpad["execution_mode"] = "stub"

    return {"node_history": state.node_history + ["execute"]}


# ═══════════════════════════════════════════════════════════════════
# Node 5: Parse output
# ═══════════════════════════════════════════════════════════════════

def parse_output(state) -> dict:
    """Parse Gaussian .log for energy, convergence, errors.

    Extracts:
    - SCF energy (final single-point)
    - Convergence status (normal termination check)
    - Imaginary frequencies (for freq jobs)
    - Wall time

    TODO: domain developer implements actual log parsing logic.
    """
    task = state.task
    params: GaussianParams = task.params

    # Stub result — domain developer replaces with actual parsing
    result_data = GaussianResult(
        energy=None,       # parse from "SCF Done:" line
        dipole=None,       # parse from dipole moment section
        n_imag_freq=0,     # parse from frequency section
        converged=False,   # check for "Normal termination"
        checkpoint_path=None,
        wall_time_s=0.0,
    )

    # Register artifacts (stub — actual paths from real execution)
    # Domain developer registers actual produced files

    return {
        "result_data": result_data,
        "status": state.status.__class__.DONE,
        "node_history": state.node_history + ["parse_output"],
    }


# ═══════════════════════════════════════════════════════════════════
# Helpers (domain developer expands these)
# ═══════════════════════════════════════════════════════════════════

def _render_gaussian_input(params: GaussianParams, task_id: str) -> str:
    """Render .gjf content from params. Stub — domain dev implements."""
    from agents.gaussian.prompts import GAUSSIAN_OPT_TEMPLATE

    additional_kw = " ".join(
        f"{k}={v}" for k, v in params.additional_keywords.items()
    )

    return GAUSSIAN_OPT_TEMPLATE.format(
        memory_mb=params.memory_mb,
        nprocs=params.nprocs,
        chk_name=task_id,
        method=params.method,
        basis=params.basis,
        job_type=params.job_type,
        additional_kw=additional_kw,
        title=task_id,
        charge=params.charge,
        spin=params.spin,
        geometry="",  # TODO: domain dev provides geometry
    )
