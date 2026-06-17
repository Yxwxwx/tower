# Agent Contract — Tower Multi-Agent System

**Version:** 1.0 · **Status:** Frozen · **Date:** 2026-06-17

This document defines the frozen interface for each agent. Every agent developer MUST implement exactly this contract. No additional implicit interfaces.

---

## 1. Universal Agent Interface

Every agent MUST:

1. Compile to a LangGraph `CompiledStateGraph`.
2. Accept `AgentTask[T]` as input (via state).
3. Return `AgentResult[T]` as output (via state).
4. Register its `RetryPolicy` with the supervisor.
5. Declare its upstream `dependencies`.
6. Access artifacts only through `ArtifactResolver`.
7. Access MCP tools only through the shared MCP client.
8. Write a heartbeat to `RunStateStore` every `heartbeat_interval_s` while RUNNING.

### 1.1 Base State

```python
from tower_agent_kit.base import BaseAgentState

class BaseAgentState(BaseModel):
    trace_id: str
    task: AgentTask
    result: AgentResult | None = None
    status: TaskStatus = TaskStatus.PENDING
    node_history: list[str] = Field(default_factory=list)
```

All agent states inherit from this. Agents MAY add private fields.

### 1.2 Registration Shape

```python
# Every agent exposes this at module level
def register() -> AgentRegistration:
    return AgentRegistration(
        name="gaussian",                     # AgentName
        subgraph=gaussian_graph,             # CompiledStateGraph
        retry_policy=RetryPolicy(...),
        timeout_s=3600,
        dependencies=set(),                  # {AgentName, ...}
    )
```

---

## 2. GaussianAgent

### Contract

| Field | Value |
|-------|-------|
| `name` | `"gaussian"` |
| `dependencies` | `set()` (no upstream) |
| `retry_policy` | `RetryPolicy(is_idempotent=True, max_retries=2)` |

### Input

```python
AgentTask[GaussianParams]

class GaussianParams(BaseModel):
    method: str = "B3LYP"
    basis: str = "def2SVP"
    charge: int = 0
    spin: int = 1
    job_type: Literal["opt", "opt+freq", "sp", "irc"] = "opt"
    additional_keywords: dict[str, str] = Field(default_factory=dict)
    memory_mb: int = 4000
    nprocs: int = 8
    checkpoint_from: str | None = None
```

### Output

```python
AgentResult[GaussianResult]

class GaussianResult(BaseModel):
    energy: float | None = None
    dipole: list[float] | None = None
    n_imag_freq: int = 0
    converged: bool = False
    checkpoint_path: str | None = None
    wall_time_s: float = 0.0
```

### Must-Implement Nodes

| Node | Responsibility |
|------|---------------|
| `write_input` | Render `.gjf` from `GaussianParams` via `infra-mcp` template tool. Write to `jobs/{task_id}/`. |
| `generate_slurm_template` | Render rough Slurm script with `#SBATCH` directives for mem/nprocs/walltime. |
| `run_or_submit` | If running locally: execute Gaussian. If using HPC: produce artifact and hand off to HPC agent. |

### Artifacts Produced

| Artifact | Type | Description |
|----------|------|-------------|
| `{task_id}.gjf` | `gjf` | Gaussian input file |
| `{task_id}_opt.fchk` | `fchk` | Converged wavefunction checkpoint |
| `{task_id}.log` | `log` | Gaussian output log |
| `{task_id}_rough.sh` | `slurm` | Rough Slurm template for HPC refinement |

### Tools Required

| MCP Server | Tool | Purpose |
|-----------|------|---------|
| `infra-mcp` | `filesystem.write` | Write .gjf input |
| `infra-mcp` | `template.render` | Render Gaussian input from Jinja2 template |

---

## 3. PySCFAgent

### Contract

| Field | Value |
|-------|-------|
| `name` | `"pyscf"` |
| `dependencies` | `{"gaussian"}` |
| `retry_policy` | `RetryPolicy(is_idempotent=True, max_retries=2)` |

### Input

```python
AgentTask[PySCFParams]

class PySCFParams(BaseModel):
    fchk_path: str
    n_active_electrons: int
    n_active_orbitals: int
    basis: str = "def2SVP"
    charge: int = 0
    spin: int = 1
    cas_irrep: str | None = None
    max_memory_mb: int = 8000
```

### Output

```python
AgentResult[PySCFResult]

class PySCFResult(BaseModel):
    active_orbitals: list[int] = Field(default_factory=list)
    casscf_energy: float | None = None
    natural_occupations: list[float] = Field(default_factory=list)
    orbital_info_path: str | None = None
```

### Must-Implement Nodes

| Node | Responsibility |
|------|---------------|
| `read_fchk` | Read Gaussian checkpoint via `ArtifactResolver`. Parse MO coefficients. |
| `select_orbitals` | Select active space orbitals. Write `active_orbitals.json`. |
| `run_casscf` | Execute CASSCF via PySCF. Write `casscf_energy.json`. |
| `generate_slurm_template` | Rough Slurm for HPC. |

### Artifacts Produced

| Artifact | Type | Description |
|----------|------|-------------|
| `active_orbitals.json` | `json` | Active orbital indices + irreps |
| `casscf_energy.json` | `json` | CASSCF energy + CI coefficients |

### Tools Required

| MCP Server | Tool | Purpose |
|-----------|------|---------|
| `infra-mcp` | `filesystem.read` | Read .fchk checkpoint |
| `infra-mcp` | `filesystem.write` | Write .json output |
| `infra-mcp` | `template.render` | Render PySCF Python script |

---

## 4. OrcaAgent

### Contract

| Field | Value |
|-------|-------|
| `name` | `"orca"` |
| `dependencies` | `{"pyscf"}` |
| `retry_policy` | `RetryPolicy(is_idempotent=True, max_retries=2)` |

### Input

```python
AgentTask[OrcaParams]

class OrcaParams(BaseModel):
    orbital_info_path: str
    method: Literal["NEVPT2", "DLPNO-CCSD(T)", "CCSD(T)"] = "NEVPT2"
    basis: str = "def2-TZVP"
    charge: int = 0
    spin: int = 1
    memory_mb: int = 16000
    nprocs: int = 16
```

### Output

```python
AgentResult[OrcaResult]

class OrcaResult(BaseModel):
    energy: float | None = None
    energy_correction: float | None = None
    converged: bool = False
    log_path: str | None = None
    wall_time_s: float = 0.0
```

### Must-Implement Nodes

| Node | Responsibility |
|------|---------------|
| `read_orbitals` | Read `active_orbitals.json` via `ArtifactResolver`. |
| `write_input` | Render Orca `.inp` from `OrcaParams` + orbital info. |
| `generate_slurm_template` | Rough Slurm for HPC. |
| `run_or_submit` | Execute or delegate to HPC. |

### Artifacts Produced

| Artifact | Type | Description |
|----------|------|-------------|
| `{task_id}.inp` | `inp` | Orca input file |
| `{task_id}.log` | `log` | Orca output log |
| `{task_id}_rough.sh` | `slurm` | Rough Slurm template |

### Tools Required

| MCP Server | Tool | Purpose |
|-----------|------|---------|
| `infra-mcp` | `filesystem.read` | Read orbital json |
| `infra-mcp` | `filesystem.write` | Write .inp |
| `infra-mcp` | `template.render` | Render Orca input template |

---

## 5. HPCAgent (Infrastructure)

### Contract

| Field | Value |
|-------|-------|
| `name` | `"hpc"` |
| `dependencies` | `set()` (no chemical dependencies; dispatched in parallel) |
| `retry_policy` | `RetryPolicy(is_idempotent=False, max_retries=2, requires_cleanup_before_retry=True, cleanup_steps=["scancel old job", "remove stale slurm script"])` |

### Input

```python
AgentTask[HPCParams]

class JobRequest(BaseModel):
    agent: AgentName
    input_file: str
    rough_slurm: str
    mem_per_cpu_mb: int
    nprocs: int
    walltime_hours: int

class HPCParams(BaseModel):
    jobs: list[JobRequest] = Field(default_factory=list)
    partition: str = "compute"
    account: str = "default"
    email_on_fail: str | None = None
```

### Output

```python
AgentResult[HPCResult]

class HPCResult(BaseModel):
    job_ids: dict[str, str] = Field(default_factory=dict)
    submitted_at: datetime | None = None
    node_assignment: dict[str, str] = Field(default_factory=dict)
    final_slurm_paths: dict[str, str] = Field(default_factory=dict)
```

### Must-Implement Nodes

| Node | Responsibility |
|------|---------------|
| `collect_needs` | Read `JobRequest` list. Validate all input files exist (via `ArtifactResolver`). |
| `query_resources` | Call `hpc-mcp queue-status` to check node availability, queue depth. |
| `refine_slurm` | Merge rough Slurm templates with cluster resource info. Add partition, account, node constraints. |
| `submit` | Call `sbatch` (or equivalent) via MCP. Record `job_ids`. |

### Artifacts Produced

| Artifact | Type | Description |
|----------|------|-------------|
| `{agent}_final.sh` | `slurm` | Final Slurm script for each agent job |
| `job_ids.json` | `json` | Map of agent → Slurm job ID |

### Tools Required

| MCP Server | Tool | Purpose |
|-----------|------|---------|
| `infra-mcp` | `slurm-gen.render` | Generate final Slurm scripts |
| `hpc-mcp` | `queue-status.query` | Query cluster queue/node state |

### Cleanup (Before Retry)

1. `scancel {job_id}` for all jobs in current batch.
2. Remove stale Slurm scripts.
3. Re-query cluster state before re-submit.

---

## 6. MonitorAgent (Infrastructure)

### Contract

| Field | Value |
|-------|-------|
| `name` | `"monitor"` |
| `dependencies` | `set()` (event-driven, independent) |
| `retry_policy` | `RetryPolicy(is_idempotent=True, max_retries=2)` |

### Input

```python
AgentTask[MonitorParams]

class MonitorParams(BaseModel):
    watchlist: dict[str, str] = Field(default_factory=dict)
    # {"12345": "gaussian", "12346": "orca"}
    poll_interval_s: int = 60
    max_watch_hours: int = 48
```

### Output

```python
AgentResult[MonitorResult]

class MonitorEvent(BaseModel):
    id: int = 0
    job_id: str
    agent: AgentName
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal["JOB_STARTED", "JOB_DONE", "JOB_FAILED",
                        "JOB_TIMEOUT", "JOB_OOM", "AGENT_HEARTBEAT_LOST"]
    log_snippet: str = ""
    error_category: Literal["scf_not_converged", "oom", "mpi_error",
                            "timeout", "unknown", ""] = ""
    suggestion: str = ""

class MonitorResult(BaseModel):
    events: list[MonitorEvent] = Field(default_factory=list)
    failed_jobs: list[str] = Field(default_factory=list)
    completed_jobs: list[str] = Field(default_factory=list)
    summary: str = ""
```

### Must-Implement Nodes

| Node | Responsibility |
|------|---------------|
| `poll_jobs` | Query `hpc-mcp queue-status` for each job in `watchlist`. |
| `parse_logs` | If job FAILED/TIMEOUT/OOM: fetch log via `hpc-mcp log-parser`. |
| `classify_error` | Parse log → `error_category` + `suggestion`. |
| `append_events` | Write `MonitorEvent` records to `RunStateStore.event_log` (append-only). |

### Events Written to RunStateStore

Monitor writes events; it NEVER modifies agent state directly.

| Event | When |
|-------|------|
| `JOB_STARTED` | Job transitions QUEUED → RUNNING |
| `JOB_DONE` | Job exits with code 0 |
| `JOB_FAILED` | Job exits with non-zero code |
| `JOB_TIMEOUT` | Job exceeds walltime limit |
| `JOB_OOM` | Job killed by OOM killer |
| `AGENT_HEARTBEAT_LOST` | Agent hasn't written heartbeat in `heartbeat_timeout_s` |

### Tools Required

| MCP Server | Tool | Purpose |
|-----------|------|---------|
| `hpc-mcp` | `queue-status.query` | Check job status (squeue/sacct) |
| `hpc-mcp` | `log-parser.fetch` | Fetch job stdout/stderr |
| `hpc-mcp` | `log-parser.classify` | Classify error from log patterns |

---

## 7. Agent Development Checklist

When implementing a new agent, you MUST:

- [ ] Inherit from `BaseAgentState` (from `tower-agent-kit`).
- [ ] Implement all nodes listed in your agent's "Must-Implement Nodes" section.
- [ ] Return `AgentResult[T]` with correct `status` and `artifacts_out`.
- [ ] Register all produced artifacts via `ArtifactRegistry`.
- [ ] Read all input artifacts via `ArtifactResolver` (never raw path).
- [ ] Access MCP tools through the shared MCP client.
- [ ] Write heartbeat to `RunStateStore` every 5 minutes while RUNNING.
- [ ] Expose `register() → AgentRegistration` at module level.
- [ ] Include `pyproject.toml` with dependencies on `tower-core`, `tower-contracts`, `tower-agent-kit`.
- [ ] Add a `README.md` in the agent directory explaining what it does.

If any of these are missing, the supervisor WILL NOT dispatch to your agent.

---

## 8. Error Handling Contract

### 8.1 Agent Errors

When an agent encounters an error it cannot recover from internally:

```python
return AgentResult(
    task_id=task.task_id,
    trace_id=task.trace_id,
    status=TaskStatus.FAILED,
    agent="gaussian",
    errors=["SCF not converged after 128 iterations"],
    next_action="increase SCF maxcycle or adjust damping",
)
```

### 8.2 Agent Must NOT

- ❌ Call `sys.exit()` or raise unhandled exceptions. Always return `AgentResult`.
- ❌ Modify another agent's artifacts.
- ❌ Directly modify `RunState.status` (supervisor only).
- ❌ Spawn threads or subprocesses that outlive the agent invocation.
- ❌ Write to paths outside `jobs/{task_id}/`.

---

## 9. Contract Versioning

This is `agent_contract.md` v1.0. When a breaking change is needed:

1. Create `agent_contract_v2.md`.
2. Bump `AgentTask.schema_version` to `"v2"`.
3. Supervisor routes v1 tasks to v1 agents, v2 tasks to v2 agents.
4. v1 agents continue to work until all agents have migrated.
