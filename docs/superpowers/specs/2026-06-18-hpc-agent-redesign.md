# HPC Agent Redesign Spec

**Date:** 2026-06-18
**Status:** Approved

## 1. Problem

Current HPC agent is entirely mock: fake node lists, template Slurm scripts with placeholder commands, UUID-based job IDs. The compute pipeline requires real script execution — logs from domain agents must flow back for parsing.

## 2. Design

### 2.1 Topology

HPC agent has a single pre-computation path (no post — HPC does not parse computation output):

```
analyze_artifacts → generate_slurm → submit → pre_done
```

### 2.2 Nodes

| Node | Purpose |
|------|---------|
| `analyze_artifacts` | Read upstream artifacts (script, run_command, resources). Run `module avail`, check `/nvme` existence. LLM analyzes: what modules/conda env are needed, what partition fits the resource request. |
| `generate_slurm` | LLM generates a complete, submission-ready Slurm script using the user's template pattern (TMPDIR on /nvme, module load, conda activate, OMP_NUM_THREADS, tar pack-back). |
| `submit` | Run `sbatch <script>`. If sbatch unavailable, fallback to `bash <script> &` and capture PID. Return real job_id. |
| `pre_done` | Store generated Slurm script + job_id as artifacts. |

### 2.3 Artifact Flow

Input (from domain agent): script artifact + slurm artifact + run_command artifact + resources dict

Output: slurm_script artifact + job_id + log artifact reference

## 3. Contract Changes

### 3.1 HPCParams — add cluster info overrides

```python
class HPCParams(BaseModel):
    jobs: list[JobRequest] = Field(default_factory=list)
    partition: str = "compute"
    account: str = "default"
    email_on_fail: str | None = None
    # New: explicit paths from supervisor
    nvme_path: str = "/nvme"
    scratch_path: str = ""  # optional scratch dir
```

### 3.2 HPCResult — add real execution fields

```python
class HPCResult(BaseModel):
    job_ids: dict[str, str] = Field(default_factory=dict)
    job_state: str = ""           # "RUNNING"|"COMPLETED"|"FAILED"
    slurm_artifact_id: str = ""   # generated Slurm script
    log_artifact_id: str = ""     # computation log
    tmpdir_artifact_id: str = ""  # tarball of TMPDIR
    submitted_at: str = ""
    node_assignment: dict = Field(default_factory=dict)
```

### 3.3 PySCFParams — add resources

```python
class PySCFParams(BaseModel):
    task_description: str = ""
    fchk_artifact_id: str = ""
    charge: int = 0
    spin: int = 0
    additional_info: dict = Field(default_factory=dict)
    resources: dict = Field(default_factory=dict)
    # {"omp_threads": 8, "memory_mb": 20000, "walltime_hours": 24,
    #  "modules": [...], "conda_env": "...", "partition_hint": "cpu32"}
```

## 4. Prompt Files

Located at `prompts/hpc/`:

| File | Purpose |
|------|---------|
| `hpc_analyze.md` | Analyze upstream artifacts + cluster state → resource plan |
| `hpc_generate_slurm.md` | Generate complete Slurm script from resource plan |

## 5. Files Changed

| File | Change |
|------|--------|
| `prompts/hpc/hpc_analyze.md` | New |
| `prompts/hpc/hpc_generate_slurm.md` | New |
| `contracts/src/contracts/hpc_task.py` | Rewrite Params and Result |
| `contracts/src/contracts/pyscf_task.py` | Add `resources` field to Params |
| `agents/hpc/agent.py` | Rewrite with new topology |
| `agents/hpc/nodes.py` | New — LLM-powered execution nodes |
| `agents/pyscf/nodes.py` | `generate_input` now returns `resources` |
| `prompts/pyscf/pyscf_generate.md` | Add resources to output format |
| `agents/supervisor/agent.py` | Update `_build_hpc_params` to pass resources |

## 6. Slurm Template (canonical pattern)

```
#!/bin/bash
#SBATCH --job-name {job_name}
#SBATCH --nodes=1
#SBATCH --time={walltime}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={omp_threads}
#SBATCH --partition={partition}
#SBATCH --mem={memory_mb}M
#SBATCH --no-requeue

TMPDIR=/nvme/$USER/{job_name}
if [ -d $TMPDIR ]; then rm -rf $TMPDIR; fi
mkdir -p $TMPDIR

module purge
{module_loads}
conda activate {conda_env}

export PYSCF_TMPDIR=$TMPDIR
export PYSCF_MAX_MEMORY={memory_mb}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

{run_command}

mkdir -p "${PWD}/tmp"
tar cf - -C "$TMPDIR" . | tar xf - -C "${PWD}/tmp"
rm -rf "$TMPDIR"
```
