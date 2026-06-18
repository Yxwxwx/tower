"""HPC agent nodes — analyze artifacts, generate Slurm, submit.

Pipeline:
    analyze_artifacts → generate_slurm → submit → pre_done

All prompts loaded from prompts/hpc/*.md at module load time.
"""
import json
import os
import subprocess
import uuid
from pathlib import Path

from contracts.hpc_task import HPCResult
from contracts.agent_task import Artifact, TaskStatus

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "hpc"


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    return path.read_text() if path.exists() else ""


HPC_ANALYZE_PROMPT = _load_prompt("hpc_analyze.md")
HPC_GENERATE_SLURM_PROMPT = _load_prompt("hpc_generate_slurm.md")


# ═══════════════════════════════════════════════════════════════════
# Cluster query tools
# ═══════════════════════════════════════════════════════════════════


def _query_modules() -> str:
    """Query available modules from the cluster. Sources module environment first."""
    try:
        result = subprocess.run(
            "bash -c 'source /etc/profile.d/modules.sh 2>/dev/null; module avail 2>&1'",
            shell=True, capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip()[:5000]
    except Exception:
        return ""


def _check_nvme(path: str) -> dict:
    """Check NVMe/scratch disk availability and space."""
    info = {"exists": False, "path": path, "free_gb": 0, "total_gb": 0}
    if not path or not os.path.isdir(path):
        return info
    try:
        stat = os.statvfs(path)
        info["exists"] = True
        info["free_gb"] = (stat.f_bavail * stat.f_frsize) // (1024 ** 3)
        info["total_gb"] = (stat.f_blocks * stat.f_frsize) // (1024 ** 3)
    except Exception:
        pass
    return info


def _query_partitions() -> str:
    """Query available Slurm partitions."""
    try:
        result = subprocess.run(
            "sinfo -o '%P|%c|%m|%a|%D' 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:2000]
    except Exception:
        pass
    return ""


def _update_scratchpad(state, **kwargs) -> dict:
    return {"scratchpad": {**state.scratchpad, **kwargs}}


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text)


# ═══════════════════════════════════════════════════════════════════
# Analyze upstream artifacts
# ═══════════════════════════════════════════════════════════════════


def analyze_artifacts(state) -> dict:
    """Read domain agent artifacts and cluster state, produce resource plan."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from tower.llm import get_model

    task = state.task
    params = task.params

    # Extract info from upstream artifacts + HPCParams
    run_command = getattr(params, "run_command", "") or ""
    resources = {}
    agent_name = "unknown"

    for job_req in params.jobs if params else []:
        agent_name = job_req.agent
        if job_req.resources:
            resources = job_req.resources

    # Fallback: construct run_command from artifacts if not in params
    if not run_command:
        for ref in task.artifacts_in:
            if getattr(ref, "type", "") == "gjf":
                script_name = getattr(ref, "artifact_id", "").replace("-script", ".py")
                run_command = "python -u {script_name} > pyscf.log 2>&1"
                break

    # Cluster state
    nvme_path = params.nvme_path if params else "/nvme"
    nvme_available = os.path.isdir(nvme_path)

    # Query module system — module is a bash function, must source environment first
    module_list = _query_modules()
    if not module_list:
        module_list = "(module system not available)"

    partition_hint = resources.get("partition_hint", params.partition if params else "compute")

    # Cluster state
    nvme_info = _check_nvme(nvme_path)
    partitions = _query_partitions()

    # LLM analysis
    prompt = HPC_ANALYZE_PROMPT.format(
        agent_name=agent_name,
        task_description=getattr(task, "goal", ""),
        run_command=run_command,
        resources_json=json.dumps(resources, indent=2),
        module_list=module_list,
        nvme_available="yes" if nvme_available else "no",
        nvme_path=nvme_path,
        nvme_free_gb=nvme_info["free_gb"],
        partition_hint=partition_hint,
        partitions_available=partitions,
    )

    try:
        model = get_model(temperature=0.0)
        response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Analyze the resource needs."),
        ])
        data = _parse_json_response(response.content)
    except Exception:
        data = _fallback_analysis(resources, run_command, agent_name, nvme_available, nvme_path)

    return {
        "node_history": state.node_history + ["analyze_artifacts"],
        **_update_scratchpad(
            state,
            analysis=data,
            run_command=run_command,
            agent_name=agent_name,
            nvme_available=nvme_available,
            nvme_path=nvme_path,
        ),
    }


def _fallback_analysis(resources: dict, run_command: str, agent_name: str,
                       nvme_available: bool, nvme_path: str) -> dict:
    """Fallback resource analysis when LLM is unavailable."""
    job_name = f"tower_{agent_name}_{uuid.uuid4().hex[:6]}"
    return {
        "job_name": job_name,
        "omp_threads": resources.get("omp_threads", 8),
        "memory_mb": resources.get("memory_mb", 8000),
        "walltime_hours": resources.get("walltime_hours", 24),
        "partition": resources.get("partition_hint", "compute"),
        "module_loads": [f"module load {m}" for m in resources.get("modules", [])],
        "conda_env": "",
        "pythonpath": resources.get("pythonpath", ""),
        "use_nvme": nvme_available,
        "nvme_path": nvme_path,
        "tmpdir": f"{nvme_path}/$USER/{job_name}" if nvme_available else "",
        "exec_mode": "slurm",
        "submit_mode": "sbatch",
        "warnings": [],
        "env_exports": [],
    }


# ═══════════════════════════════════════════════════════════════════
# Generate Slurm script
# ═══════════════════════════════════════════════════════════════════


def generate_slurm(state) -> dict:
    """Generate a complete Slurm batch script from the resource analysis."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from tower.llm import get_model

    sp = state.scratchpad
    analysis = sp.get("analysis", {})
    task_id = state.task_id
    run_dir = task_id.rsplit("-", 1)[0] if "-" in task_id else task_id

    prompt = HPC_GENERATE_SLURM_PROMPT.format(
        job_name=analysis.get("job_name", "tower_job"),
        omp_threads=analysis.get("omp_threads", 8),
        memory_mb=analysis.get("memory_mb", 8000),
        walltime_hours=analysis.get("walltime_hours", 24),
        partition=analysis.get("partition", "compute"),
        module_loads="\n".join(analysis.get("module_loads", [])),
        conda_env="",
        pythonpath=analysis.get("pythonpath", ""),
        use_nvme="true" if analysis.get("use_nvme") else "false",
        nvme_path=analysis.get("nvme_path", "/nvme"),
        tmpdir=analysis.get("tmpdir", ""),
        run_command=sp.get("run_command", ""),
        env_exports="\n".join(analysis.get("env_exports", [])),
        exec_mode=analysis.get("exec_mode", "slurm"),
        job_dir=run_dir,
    )

    try:
        model = get_model(temperature=0.0)
        response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate the Slurm script."),
        ])
        data = _parse_json_response(response.content)
        slurm_script = data.get("slurm_script", "")
    except Exception:
        slurm_script = _fallback_slurm(analysis, sp.get("run_command", ""), run_dir)

    return {
        "node_history": state.node_history + ["generate_slurm"],
        **_update_scratchpad(state, slurm_script=slurm_script, run_dir=run_dir),
    }


def _fallback_slurm(analysis: dict, run_command: str, job_dir: str = "") -> str:
    """Fallback Slurm script when LLM is unavailable."""
    job_name = analysis.get("job_name", "tower_job")
    omp = analysis.get("omp_threads", 8)
    mem_gb = max(1, analysis.get("memory_mb", 8000) // 1024)
    wall_h = analysis.get("walltime_hours", 24)
    days = wall_h // 24
    hours = wall_h % 24
    partition = analysis.get("partition", "compute")
    tmpdir = analysis.get("tmpdir", "")
    module_lines = "\n".join(analysis.get("module_loads", []))
    out_err = ""
    cd_line = f"\ncd {run_dir}" if run_dir else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --time={days}-{hours:02d}:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={omp}
#SBATCH --partition={partition}
#SBATCH --mem={mem_gb}G
#SBATCH --no-requeue{out_err}

"""
    if tmpdir:
        script += f"""TMPDIR={tmpdir}
if [ -d $TMPDIR ]; then rm -rf $TMPDIR; fi
mkdir -p $TMPDIR

"""
    if module_lines:
        script += f"module purge\n{module_lines}\n"

    script += f"""
export PYSCF_TMPDIR=$TMPDIR
export PYSCF_MAX_MEMORY={analysis.get('memory_mb', 8000)}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{cd_line}

{run_command}

"""
    if tmpdir:
        script += 'mkdir -p "tmp"\ntar cf - -C "$TMPDIR" . | tar xf - -C "tmp"\nrm -rf "$TMPDIR"\n'
    elif tmpdir:
        script += """mkdir -p "${PWD}/tmp"
tar cf - -C "$TMPDIR" . | tar xf - -C "${PWD}/tmp"
rm -rf "$TMPDIR"
"""
    return script


# ═══════════════════════════════════════════════════════════════════
# Submit
# ═══════════════════════════════════════════════════════════════════


def submit(state) -> dict:
    """Submit the Slurm script via sbatch, or fallback to local bash."""
    sp = state.scratchpad
    task_id = state.task_id
    slurm_script = sp.get("slurm_script", "")
    analysis = sp.get("analysis", {})
    run_command = sp.get("run_command", "")

    if not slurm_script:
        return {
            "node_history": state.node_history + ["submit"],
            **_update_scratchpad(state, job_id="failed-no-script", job_state="FAILED",
                                 log_path="", submit_ok=False),
        }

    # All files in one flat directory: $PWD/run-{run_id}/
    run_dir = task_id.rsplit("-", 1)[0] if "-" in task_id else task_id
    script_dir = Path(run_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    job_name = analysis.get("job_name", "tower_job")
    script_path = script_dir / f"{job_name}.sh"
    script_path.write_text(slurm_script)
    log_path = str(script_dir / f"{analysis.get('job_name', task_id)}.log")

    submit_mode = analysis.get("submit_mode", "sbatch")
    job_id = ""
    job_state = ""

    if submit_mode == "sbatch":
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                # "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                job_state = "QUEUED"
            else:
                job_id = f"local-{uuid.uuid4().hex[:8]}"
                job_state = "RUNNING"
                # Fallback: run locally
                subprocess.Popen(
                    f"bash {script_path} > {log_path} 2>&1",
                    shell=True,
                )
        except FileNotFoundError:
            # sbatch not available — local execution
            job_id = f"local-{uuid.uuid4().hex[:8]}"
            job_state = "RUNNING"
            log_path = str(script_dir / "pyscf.log")
            subprocess.Popen(
                f"cd {script_dir} && bash -c '{run_command}' > {log_path} 2>&1",
                shell=True,
            )
    else:
        # Direct local execution
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        job_state = "RUNNING"
        log_path = str(script_dir / "pyscf.log")
        subprocess.Popen(
            f"bash -c '{run_command}' > {log_path} 2>&1",
            shell=True,
        )

    return {
        "node_history": state.node_history + ["submit"],
        **_update_scratchpad(
            state,
            job_id=job_id,
            job_state=job_state,
            log_path=log_path,
            slurm_path=str(script_path),
            submit_ok=True,
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════


def pre_done(state) -> dict:
    """Return job results as artifacts to supervisor."""
    sp = state.scratchpad
    task_id = state.task_id
    job_id = sp.get("job_id", "unknown")
    log_path = sp.get("log_path", "")
    slurm_path = sp.get("slurm_path", "")

    log_artifact_id = f"{task_id}-log-{job_id}"

    return {
        "status": TaskStatus.DONE,
        "result_data": HPCResult(
            job_ids={sp.get("agent_name", "unknown"): job_id},
            job_state=sp.get("job_state", "COMPLETED"),
            slurm_artifact_id=f"{task_id}-slurm",
            log_artifact_id=log_artifact_id,
            submitted_at="",
        ),
        "artifacts_out": [
            Artifact(
                artifact_id=f"{task_id}-slurm",
                path=slurm_path,
                type="slurm",
                description=f"Slurm script for {job_id}",
                producer_agent="hpc",
                producer_task_id=task_id,
            ),
            Artifact(
                artifact_id=log_artifact_id,
                path=log_path,
                type="log",
                description=f"Computation log for job {job_id}",
                producer_agent="hpc",
                producer_task_id=task_id,
            ),
        ],
        "node_history": state.node_history + ["pre_done"],
    }
