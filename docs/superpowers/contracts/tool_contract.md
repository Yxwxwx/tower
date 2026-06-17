# Tool Contract — Tower Multi-Agent System

**Version:** 1.0 · **Status:** Frozen · **Date:** 2026-06-17

This document defines the frozen interface for all MCP tools in the shared tool layer. Tools are exposed through exactly two MCP servers: `infra-mcp` and `hpc-mcp`.

---

## 1. MCP Server Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP CLIENTS                          │
│  (gaussian, pyscf, orca, hpc, monitor agents)          │
└────────┬──────────────────────────────────┬─────────────┘
         │                                  │
    ┌────▼──────────┐              ┌────────▼────────┐
    │  infra-mcp    │              │   hpc-mcp       │
    │  (stdio)      │              │   (stdio)       │
    │               │              │                 │
    │ filesystem.*  │              │ queue-status.*  │
    │ template.*    │              │ log-parser.*    │
    │ slurm-gen.*   │              │                 │
    └───────────────┘              └─────────────────┘
```

**Both servers use `stdio` transport.** Each is a separate process, launched and managed by the MCP client manager in `tower-core`.

---

## 2. infra-mcp

### Server Config

```yaml
name: infra-mcp
transport: stdio
command: ["python", "-m", "mcp_servers.infra_mcp.server"]
```

### 2.1 `filesystem.read`

Read a file from the job workspace.

```json
// Input
{
  "path": "jobs/n2-nevpt2-gaussian-001/N2_opt.fchk",
  "max_bytes": 10485760
}

// Output (success)
{
  "content": "...",
  "path": "jobs/n2-nevpt2-gaussian-001/N2_opt.fchk",
  "size_bytes": 2048576,
  "content_hash": "sha256:abc123..."
}

// Output (error)
{
  "error": "File not found: jobs/.../N2_opt.fchk",
  "error_code": "NOT_FOUND"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | Yes |
| **Timeout** | 10s |
| **Failure mode** | Returns error dict; caller retries |

---

### 2.2 `filesystem.write`

Write a file to the job workspace. Creates parent directories.

```json
// Input
{
  "path": "jobs/n2-nevpt2-gaussian-001/N2_opt.gjf",
  "content": "%mem=4GB\n%nproc=8\n#p B3LYP/def2SVP opt\n\n..."
}

// Output (success)
{
  "ok": true,
  "path": "jobs/n2-nevpt2-gaussian-001/N2_opt.gjf",
  "size_bytes": 1234,
  "content_hash": "sha256:def456..."
}

// Output (error)
{
  "error": "Permission denied: /etc/passwd",
  "error_code": "PERMISSION_DENIED"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | No (overwrites). Caller must ensure unique path. |
| **Timeout** | 10s |
| **Failure mode** | Returns error dict. Retry with new path. |
| **Safety** | Rejects paths outside project root and `jobs/` directory. |

---

### 2.3 `template.render`

Render a Jinja2 template with parameters.

```json
// Input
{
  "template_name": "gaussian_opt.gjf.j2",
  "params": {
    "method": "B3LYP",
    "basis": "def2SVP",
    "charge": 0,
    "spin": 1,
    "memory_mb": 4000,
    "nprocs": 8
  }
}

// Output (success)
{
  "rendered": "%mem=4GB\n%nproc=8\n#p B3LYP/def2SVP opt\n\nN2 optimization\n\n0 1\nN 0.0 0.0 0.0\nN 0.0 0.0 1.1\n\n",
  "template_name": "gaussian_opt.gjf.j2"
}

// Output (error)
{
  "error": "Template 'orca_unknown.inp.j2' not found",
  "error_code": "TEMPLATE_NOT_FOUND"
}
```

**Available templates (v1):**

| Template | For Agent |
|----------|-----------|
| `gaussian_opt.gjf.j2` | GaussianAgent |
| `gaussian_freq.gjf.j2` | GaussianAgent |
| `pyscf_casscf.py.j2` | PySCFAgent |
| `orca_nevpt2.inp.j2` | OrcaAgent |
| `orca_dlpno.inp.j2` | OrcaAgent |

| Property | Value |
|----------|-------|
| **Idempotent** | Yes (pure function) |
| **Timeout** | 5s |
| **Failure mode** | Returns error dict for unknown template or missing params. |

---

### 2.4 `slurm-gen.render`

Generate a full Slurm script from a rough template + cluster resource info.

```json
// Input
{
  "rough_template": "#!/bin/bash\n#SBATCH --mem={mem_mb}\n#SBATCH --cpus-per-task={nprocs}\n{srun_command}",
  "params": {
    "job_name": "n2-nevpt2-gaussian",
    "mem_per_cpu_mb": 4000,
    "nprocs": 8,
    "walltime_hours": 24,
    "partition": "compute",
    "account": "default",
    "node_list": "node03,node04",
    "srun_command": "g16 < N2_opt.gjf > N2_opt.log",
    "email_on_fail": "user@example.com"
  }
}

// Output (success)
{
  "script": "#!/bin/bash\n#SBATCH --job-name=n2-nevpt2-gaussian\n...",
  "job_name": "n2-nevpt2-gaussian"
}

// Output (error)
{
  "error": "Missing required param: partition",
  "error_code": "MISSING_PARAM"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | Yes (pure function) |
| **Timeout** | 5s |
| **Failure mode** | Returns error dict. |

---

## 3. hpc-mcp

### Server Config

```yaml
name: hpc-mcp
transport: stdio
command: ["python", "-m", "mcp_servers.hpc_mcp.server"]
```

### 3.1 `queue-status.query`

Query cluster queue and node status. This is a **read-only** tool.

```json
// Input
{
  "query_type": "job",
  "job_id": "12345"
}

// Output (success, job running)
{
  "job_id": "12345",
  "status": "RUNNING",
  "node": "node03",
  "started_at": "2026-06-17T10:30:00",
  "elapsed_s": 3600,
  "exit_code": null
}

// Output (success, job done)
{
  "job_id": "12345",
  "status": "COMPLETED",
  "node": "node03",
  "started_at": "2026-06-17T10:30:00",
  "elapsed_s": 7200,
  "exit_code": 0
}

// Input (query cluster-wide)
{
  "query_type": "cluster",
  "partition": "compute"
}

// Output
{
  "nodes": [
    {"name": "node01", "state": "idle", "cpus_total": 64, "cpus_avail": 64, "mem_total_mb": 256000},
    {"name": "node02", "state": "mixed", "cpus_total": 64, "cpus_avail": 16, "mem_total_mb": 256000}
  ],
  "queue_depth": 12,
  "partition": "compute"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | Yes (read-only) |
| **Timeout** | 15s (cluster commands can be slow) |
| **Failure mode** | `"error": "squeue: command not found"` if Slurm not available. |
| **Rate limit** | Monitor: 1 query per 60s per job. HPCAgent: 1 query per 30s. |

### 3.2 `queue-status.submit`

Submit a job to the cluster. This is a **write** tool — NOT idempotent.

```json
// Input
{
  "slurm_script_path": "jobs/n2-nevpt2-gaussian-001/gaussian_final.sh"
}

// Output (success)
{
  "job_id": "12345",
  "submitted_at": "2026-06-17T10:25:00",
  "script_path": "jobs/n2-nevpt2-gaussian-001/gaussian_final.sh"
}

// Output (error)
{
  "error": "sbatch: error: Batch job submission failed: Partition 'gpu' not found",
  "error_code": "SUBMIT_FAILED"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | **No**. Each call submits a NEW job. Caller must track `job_id` and `scancel` old jobs before retry. |
| **Timeout** | 30s |
| **Cleanup** | `scancel {job_id}` before retry. |

### 3.3 `queue-status.cancel`

Cancel a running or queued job.

```json
// Input
{
  "job_id": "12345"
}

// Output (success)
{
  "ok": true,
  "job_id": "12345",
  "cancelled_at": "2026-06-17T11:00:00"
}

// Output (error)
{
  "error": "scancel: job 12345 not found (already completed?)",
  "error_code": "JOB_NOT_FOUND"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | Yes (cancelling an already-cancelled job is a no-op). |
| **Timeout** | 15s |

### 3.4 `log-parser.fetch`

Fetch the stdout/stderr log for a job.

```json
// Input
{
  "job_id": "12347",
  "tail_lines": 500
}

// Output (success)
{
  "job_id": "12347",
  "stdout": "...",
  "stderr": "SCF not converged niter= 128\n...",
  "exit_code": 1,
  "fetched_at": "2026-06-17T12:00:00"
}

// Output (error)
{
  "error": "Log file not found for job 12347 (job still queued?)",
  "error_code": "LOG_NOT_FOUND"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | Yes (read-only) |
| **Timeout** | 10s |
| **Failure mode** | `LOG_NOT_FOUND` if job is still queued or log path unknown. |

### 3.5 `log-parser.classify`

Classify an error from a log snippet.

```json
// Input
{
  "stdout": "...",
  "stderr": "SCF not converged niter= 128\n Largest AbiJ gradient: 0.001234\n",
  "exit_code": 1,
  "agent": "orca"
}

// Output (success, classified)
{
  "error_category": "scf_not_converged",
  "confidence": 0.95,
  "suggestion": "Increase SCF maxiter or adjust damping",
  "details": "SCF failed after 128 iterations"
}

// Output (success, unclassified)
{
  "error_category": "unknown",
  "confidence": 0.0,
  "suggestion": "",
  "details": "No known error pattern matched"
}
```

| Property | Value |
|----------|-------|
| **Idempotent** | Yes (pure function) |
| **Timeout** | 5s |
| **Known categories (v1)** | `scf_not_converged`, `oom`, `mpi_error`, `timeout`, `unknown` |
| **Extensibility** | New error patterns added via regex/LLM classification in `log-parser` server. |

---

## 4. MCP Client Usage Rules

### 4.1 All Agents

```python
# Agents call MCP tools through the shared client manager
from tower.mcp.client import get_mcp_client

client = get_mcp_client()

# Read a file
result = await client.call_tool("infra-mcp", "filesystem.read", {
    "path": artifact_path,
})

# Render a template
result = await client.call_tool("infra-mcp", "template.render", {
    "template_name": "gaussian_opt.gjf.j2",
    "params": {...},
})
```

### 4.2 Tool Invocation Rules

| Rule | Description |
|------|-------------|
| **All tools return a dict** | `{"ok": ...}` or `{"error": ..., "error_code": ...}` |
| **Check error field** | Always check `"error" in result` before using result data. |
| **Timeout handling** | MCP client enforces per-tool timeout. On timeout, the call raises `MCPTimeoutError`. |
| **Retry idempotent tools** | If a read-only tool times out, retry once after 5s. |
| **Never retry non-idempotent tools** | E.g., `queue-status.submit` — retry creates duplicate jobs. |
| **Respect rate limits** | `queue-status.query` has per-agent rate limits. |

### 4.3 Artifact References vs Raw Paths

Agents MUST NOT hardcode file paths. When an agent needs to read an artifact produced by another agent:

```python
# CORRECT: resolve by artifact_id
record = await artifact_resolver.resolve(artifact_id)
content = await client.call_tool("infra-mcp", "filesystem.read", {
    "path": record.path,
})

# WRONG: hardcode path
content = await client.call_tool("infra-mcp", "filesystem.read", {
    "path": "output/N2_opt.fchk",  # ← FORBIDDEN
})
```

---

## 5. Adding a New Tool

To add a new shared tool:

1. Implement it in the appropriate MCP server (`infra-mcp` or `hpc-mcp`).
2. Add its schema to this document (new section under the server).
3. Mark idempotency, timeout, and failure mode.
4. Update `AgentContract` if the tool is required by a specific agent.
5. Bump the minor version of `tool_contract.md` (e.g., 1.0 → 1.1).

### When to create a new MCP server

Do NOT create a new MCP server unless:
- The tool category is fundamentally different from filesystem/template/slurm (infra) and queue/log (hpc).
- The tool requires a separate process lifecycle or different security boundary.

Start with the 2 existing servers. Split when there are 5+ unrelated tools in one server.

---

## 6. Tool Contract Summary

| Server | Tool | Idempotent | Timeout | Side Effects |
|--------|------|-----------|---------|-------------|
| `infra-mcp` | `filesystem.read` | ✅ | 10s | None |
| `infra-mcp` | `filesystem.write` | ❌ | 10s | Creates/overwrites file |
| `infra-mcp` | `template.render` | ✅ | 5s | None |
| `infra-mcp` | `slurm-gen.render` | ✅ | 5s | None |
| `hpc-mcp` | `queue-status.query` | ✅ | 15s | None |
| `hpc-mcp` | `queue-status.submit` | ❌ | 30s | Creates new Slurm job |
| `hpc-mcp` | `queue-status.cancel` | ✅ | 15s | Cancels/kills Slurm job |
| `hpc-mcp` | `log-parser.fetch` | ✅ | 10s | None |
| `hpc-mcp` | `log-parser.classify` | ✅ | 5s | None |

---

## 7. Error Codes (All Tools)

| Error Code | Meaning | Caller Action |
|-----------|---------|---------------|
| `NOT_FOUND` | File/job/template not found | Verify path/ID, retry once |
| `PERMISSION_DENIED` | Cannot read/write path | Check path, escalate |
| `TEMPLATE_NOT_FOUND` | Template name unknown | Verify template name |
| `MISSING_PARAM` | Required param not provided | Fix input, retry |
| `SUBMIT_FAILED` | Slurm rejected the job | Read error details, adjust params |
| `JOB_NOT_FOUND` | job_id not in Slurm | Job may have completed; ignore |
| `LOG_NOT_FOUND` | Log file not available yet | Wait, retry in next poll cycle |
| `TIMEOUT` | Tool exceeded time limit | Retry once (idempotent tools only) |
| `UNKNOWN` | Unclassified error | Escalate to supervisor |
