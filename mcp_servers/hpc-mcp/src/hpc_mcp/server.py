"""HPC MCP server — queue status, job submission, log parsing.

Exposes tools per docs/superpowers/contracts/tool_contract.md:
- queue-status.query: query cluster queue and node status (READ-ONLY)
- queue-status.submit: submit job to cluster (NOT idempotent)
- queue-status.cancel: cancel a running/queued job
- log-parser.fetch: fetch stdout/stderr for a job
- log-parser.classify: classify error from log snippet

Transport: stdio
"""


# ═══════════════════════════════════════════════════════════════════
# Tool: queue-status.query
# Idempotent: Yes (read-only) | Timeout: 15s
# ═══════════════════════════════════════════════════════════════════

async def queue_status_query(query_type: str, **kwargs) -> dict:
    """Query cluster queue and node status.

    query_type="job": requires job_id. Returns job status, node, elapsed time.
    query_type="cluster": requires partition. Returns node list, queue depth.

    TODO: domain developer implements squeue/sinfo/sacct parsing.
    """
    return {"stub": True, "query_type": query_type, "message": "HPC tools not yet configured"}


# ═══════════════════════════════════════════════════════════════════
# Tool: queue-status.submit
# Idempotent: NO (each call = new job) | Timeout: 30s
# ═══════════════════════════════════════════════════════════════════

async def queue_status_submit(slurm_script_path: str) -> dict:
    """Submit a job via sbatch.

    WARNING: NOT idempotent. Caller must track job_id and scancel old jobs
    before retry.

    TODO: domain developer implements sbatch submission.
    """
    return {
        "stub": True,
        "job_id": "stub-00000",
        "submitted_at": "2026-06-17T00:00:00",
        "script_path": slurm_script_path,
    }


# ═══════════════════════════════════════════════════════════════════
# Tool: queue-status.cancel
# Idempotent: Yes | Timeout: 15s
# ═══════════════════════════════════════════════════════════════════

async def queue_status_cancel(job_id: str) -> dict:
    """Cancel a running or queued job via scancel."""
    return {"ok": True, "job_id": job_id, "stub": True}


# ═══════════════════════════════════════════════════════════════════
# Tool: log-parser.fetch
# Idempotent: Yes (read-only) | Timeout: 10s
# ═══════════════════════════════════════════════════════════════════

async def log_parser_fetch(job_id: str, tail_lines: int = 500) -> dict:
    """Fetch stdout/stderr for a completed job."""
    return {
        "job_id": job_id,
        "stdout": "(stub — no real log)",
        "stderr": "",
        "exit_code": 0,
        "stub": True,
    }


# ═══════════════════════════════════════════════════════════════════
# Tool: log-parser.classify
# Idempotent: Yes (pure function) | Timeout: 5s
# ═══════════════════════════════════════════════════════════════════

async def log_parser_classify(stdout: str, stderr: str, exit_code: int, agent: str) -> dict:
    """Classify an error from log output.

    Known categories (v1): scf_not_converged, oom, mpi_error, timeout, unknown.

    TODO: domain developer implements regex/LLM classification.
    """
    return {
        "error_category": "unknown",
        "confidence": 0.0,
        "suggestion": "",
        "details": "Stub — classification not yet configured",
    }


# ═══════════════════════════════════════════════════════════════════
# Tool registry
# ═══════════════════════════════════════════════════════════════════

TOOLS = {
    "queue-status.query": queue_status_query,
    "queue-status.submit": queue_status_submit,
    "queue-status.cancel": queue_status_cancel,
    "log-parser.fetch": log_parser_fetch,
    "log-parser.classify": log_parser_classify,
}

TOOL_SCHEMAS = {
    "queue-status.query": {
        "description": "Query cluster queue and node status (read-only)",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {"type": "string", "enum": ["job", "cluster"]},
                "job_id": {"type": "string"},
                "partition": {"type": "string"},
            },
            "required": ["query_type"],
        },
    },
    "queue-status.submit": {
        "description": "Submit job to cluster via sbatch (NOT idempotent)",
        "input_schema": {
            "type": "object",
            "properties": {
                "slurm_script_path": {"type": "string"},
            },
            "required": ["slurm_script_path"],
        },
    },
    "queue-status.cancel": {
        "description": "Cancel a running/queued job via scancel",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    },
    "log-parser.fetch": {
        "description": "Fetch stdout/stderr log for a job",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "tail_lines": {"type": "integer", "default": 500},
            },
            "required": ["job_id"],
        },
    },
    "log-parser.classify": {
        "description": "Classify error from log snippet",
        "input_schema": {
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
                "agent": {"type": "string"},
            },
            "required": ["stdout", "stderr", "exit_code", "agent"],
        },
    },
}
