"""Gaussian agent nodes — pre-computation and post-computation paths.

LangGraph Pydantic state is immutable. Nodes return dict updates.
scratchpad mutations must use {**state.scratchpad, key: value} pattern.
"""
from contracts.gaussian_task import GaussianParams, GaussianResult
from contracts.agent_task import Artifact, ArtifactStatus, TaskStatus


def _update_scratchpad(state, **kwargs) -> dict:
    """Return a scratchpad update dict with new keys merged in."""
    return {"scratchpad": {**state.scratchpad, **kwargs}}


# ═══════════════════════════════════════════════════════════════════
# Pre-computation path
# ═══════════════════════════════════════════════════════════════════

def query_knowledge(state) -> dict:
    """Query long-term memory for parameters, templates, known issues.

    TODO: domain developer wires MemoryRetriever to enrich params.
    """
    return {
        "node_history": state.node_history + ["query_knowledge"],
        **_update_scratchpad(state, queried_db=True),
    }


def generate_input(state) -> dict:
    """Generate .gjf input file from GaussianParams.

    TODO: domain developer implements MCP template.render + filesystem.write.
    """
    task = state.task
    params: GaussianParams = task.params

    from agents.gaussian.prompts import GAUSSIAN_OPT_TEMPLATE
    additional_kw = " ".join(f"{k}={v}" for k, v in params.additional_keywords.items())

    content = GAUSSIAN_OPT_TEMPLATE.format(
        memory_mb=params.memory_mb, nprocs=params.nprocs,
        chk_name=task.task_id, method=params.method, basis=params.basis,
        job_type=params.job_type, additional_kw=additional_kw,
        title=task.goal, charge=params.charge, spin=params.spin,
        geometry="",
    )

    return {
        "node_history": state.node_history + ["generate_input"],
        **_update_scratchpad(state,
            gjf_content=content,
            gjf_path=f"jobs/{task.task_id}/{task.task_id}.gjf",
        ),
    }


def generate_slurm_template(state) -> dict:
    """Generate rough Slurm template for HPC refinement.

    TODO: domain developer implements MCP template.render.
    """
    task = state.task
    params: GaussianParams = task.params

    slurm = f"""#!/bin/bash
#SBATCH --job-name={task.task_id}
#SBATCH --mem={params.memory_mb}
#SBATCH --cpus-per-task={params.nprocs}
#SBATCH --time=24:00:00
# ROUGH TEMPLATE — HPC agent refines partition/account/node

g16 < {state.scratchpad.get('gjf_path', 'input.gjf')} > jobs/{task.task_id}/{task.task_id}.log
"""

    return {
        "node_history": state.node_history + ["generate_slurm_template"],
        **_update_scratchpad(state,
            slurm_content=slurm,
            slurm_path=f"jobs/{task.task_id}/{task.task_id}_rough.sh",
        ),
    }


def pre_compute_done(state) -> dict:
    """Pre-computation complete. Return artifacts to supervisor."""
    task_id = state.task_id

    return {
        "status": TaskStatus.DONE,
        "artifacts_out": [
            Artifact(artifact_id=f"{task_id}-gjf",
                     path=state.scratchpad["gjf_path"], type="gjf",
                     description="Gaussian input file",
                     producer_agent="gaussian", producer_task_id=task_id),
            Artifact(artifact_id=f"{task_id}-slurm",
                     path=state.scratchpad["slurm_path"], type="slurm",
                     description="Rough Slurm template for HPC refinement",
                     producer_agent="gaussian", producer_task_id=task_id),
        ],
        "node_history": state.node_history + ["pre_compute_done"],
    }


# ═══════════════════════════════════════════════════════════════════
# Post-computation path
# ═══════════════════════════════════════════════════════════════════

def read_output(state) -> dict:
    """Read Gaussian .log and .fchk from HPC output.

    Artifacts come via task.artifacts_in (passed by supervisor).
    """
    log_id = ""
    fchk_id = ""
    for ref in state.task.artifacts_in:
        if ref.type == "log":
            log_id = ref.artifact_id
        elif ref.type == "fchk":
            fchk_id = ref.artifact_id

    return {
        "node_history": state.node_history + ["read_output"],
        **_update_scratchpad(state, log_artifact_id=log_id, fchk_artifact_id=fchk_id),
    }


def parse_energy(state) -> dict:
    """Parse Gaussian .log for SCF energy, convergence.

    TODO: domain developer implements actual log parsing.
    """
    result_data = GaussianResult(
        energy=None,
        dipole=None,
        n_imag_freq=0,
        converged=True,
        checkpoint_path=state.scratchpad.get("fchk_artifact_id"),
        wall_time_s=0.0,
    )
    return {
        "result_data": result_data,
        "node_history": state.node_history + ["parse_energy"],
    }


def register_artifacts(state) -> dict:
    """Surface HPC-produced log/fchk as artifacts for downstream agents.

    The artifacts already exist in RunStateStore (registered by HPC/Monitor
    when the job completed). We just surface their artifact_ids so the
    supervisor can pass them to downstream agents via artifacts_in.

    No new Artifact objects are created here — the files already exist.
    """
    artifacts = []
    log_id = state.scratchpad.get("log_artifact_id", "")
    fchk_id = state.scratchpad.get("fchk_artifact_id", "")

    if log_id:
        artifacts.append(Artifact(
            artifact_id=log_id,
            path="",  # resolved at runtime via ArtifactResolver
            type="log", description="Gaussian output log (HPC-produced)",
            producer_agent="gaussian", producer_task_id=state.task_id,
            status=ArtifactStatus.READY,
        ))
    if fchk_id:
        artifacts.append(Artifact(
            artifact_id=fchk_id,
            path="",  # resolved at runtime via ArtifactResolver
            type="fchk", description="Gaussian checkpoint (HPC-produced)",
            producer_agent="gaussian", producer_task_id=state.task_id,
            status=ArtifactStatus.READY,
        ))

    return {
        "status": TaskStatus.DONE,
        "artifacts_out": artifacts,
        "node_history": state.node_history + ["register_artifacts"],
    }
