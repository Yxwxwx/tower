"""Bash 执行工具。"""
import subprocess
from langchain_core.tools import tool


@tool
def bash(command: str) -> dict:
    """Execute a bash command.
    Use for: listing files, running scripts, checking system info, searching with grep.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return {
            "command": command,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "stdout": "",
            "stderr": "Timed out after 30s",
            "returncode": -1,
            "error": "Timeout",
        }
