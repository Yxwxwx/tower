"""Infra MCP server — filesystem, template rendering, Slurm generation.

Exposes tools per docs/superpowers/contracts/tool_contract.md:
- filesystem.read: read a file from workspace
- filesystem.write: write a file to workspace
- template.render: render Jinja2 template with params
- slurm-gen.render: generate Slurm script from rough template + cluster info

Transport: stdio
"""
import json
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# Tool: filesystem.read
# Idempotent: Yes | Timeout: 10s
# ═══════════════════════════════════════════════════════════════════

async def filesystem_read(path: str, max_bytes: int = 10_485_760) -> dict:
    """Read a file from the job workspace.

    TODO: domain developer implements actual file I/O + safety checks.
    """
    # Stub implementation
    try:
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}", "error_code": "NOT_FOUND"}
        content = p.read_text()[:max_bytes]
        return {
            "content": content,
            "path": str(p),
            "size_bytes": len(content),
        }
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_code": "PERMISSION_DENIED"}
    except Exception as e:
        return {"error": str(e), "error_code": "UNKNOWN"}


# ═══════════════════════════════════════════════════════════════════
# Tool: filesystem.write
# Idempotent: No (overwrites) | Timeout: 10s
# ═══════════════════════════════════════════════════════════════════

async def filesystem_write(path: str, content: str) -> dict:
    """Write a file to the job workspace. Creates parent directories.

    Safety: rejects paths outside project root and jobs/ directory.
    """
    try:
        p = Path(path).resolve()
        # Safety check
        cwd = Path.cwd()
        if not str(p).startswith(str(cwd)):
            return {"error": f"Path outside workspace: {path}", "error_code": "PERMISSION_DENIED"}

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {
            "ok": True,
            "path": str(p),
            "size_bytes": len(content),
        }
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_code": "PERMISSION_DENIED"}
    except Exception as e:
        return {"error": str(e), "error_code": "UNKNOWN"}


# ═══════════════════════════════════════════════════════════════════
# Tool: template.render
# Idempotent: Yes (pure function) | Timeout: 5s
# ═══════════════════════════════════════════════════════════════════

async def template_render(template_name: str, params: dict) -> dict:
    """Render a Jinja2 template with params.

    Available templates (v1):
    - gaussian_opt.gjf.j2
    - gaussian_freq.gjf.j2
    - pyscf_casscf.py.j2
    - orca_nevpt2.inp.j2
    - orca_dlpno.inp.j2

    TODO: domain developer adds actual template files + Jinja2 rendering.
    """
    # Stub — returns params as JSON for now
    return {
        "rendered": json.dumps(params, indent=2),
        "template_name": template_name,
    }


# ═══════════════════════════════════════════════════════════════════
# Tool: slurm-gen.render
# Idempotent: Yes (pure function) | Timeout: 5s
# ═══════════════════════════════════════════════════════════════════

async def slurm_gen_render(rough_template: str, params: dict) -> dict:
    """Generate a full Slurm script from rough template + cluster resource info.

    TODO: domain developer implements actual Slurm rendering logic.
    """
    script = rough_template.format(**params) if "{" in rough_template else rough_template
    return {
        "script": script,
        "job_name": params.get("job_name", "tower-job"),
    }


# ═══════════════════════════════════════════════════════════════════
# Tool registry (for MCP server discovery)
# ═══════════════════════════════════════════════════════════════════

TOOLS = {
    "filesystem.read": filesystem_read,
    "filesystem.write": filesystem_write,
    "template.render": template_render,
    "slurm-gen.render": slurm_gen_render,
}

TOOL_SCHEMAS = {
    "filesystem.read": {
        "description": "Read a file from the job workspace",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "max_bytes": {"type": "integer", "default": 10485760},
            },
            "required": ["path"],
        },
    },
    "filesystem.write": {
        "description": "Write a file to the job workspace",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    "template.render": {
        "description": "Render a Jinja2 template with parameters",
        "input_schema": {
            "type": "object",
            "properties": {
                "template_name": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["template_name", "params"],
        },
    },
    "slurm-gen.render": {
        "description": "Generate a Slurm script from template + cluster info",
        "input_schema": {
            "type": "object",
            "properties": {
                "rough_template": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["rough_template", "params"],
        },
    },
}
