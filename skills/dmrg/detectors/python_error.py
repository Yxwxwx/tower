"""Python runtime error detector.

Detects Python errors in computation output (tracebacks, exceptions).
"""
import re
from tower.graph.hooks import ErrorInfo


class PythonErrorDetector:
    """Detect Python runtime errors in computation output.

    Recognizes common exception types and determines whether
    the error can be auto-fixed (MemoryError → reduce M)
    or requires human intervention (SyntaxError, ImportError).
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        combined = stdout + "\n" + stderr

        # Only check if there's a traceback or non-zero exit code
        if "Traceback (most recent call last)" not in combined and exit_code == 0:
            return None

        # Extract exception type
        exc_match = re.search(r"(\w+(?:Error|Exception|Warning))(?::|$)", combined)
        exc_type = exc_match.group(1) if exc_match else "UnknownError"

        # Extract the error message line
        msg = ""
        for line in combined.split("\n"):
            stripped = line.strip()
            if stripped.startswith(exc_type) or "Error:" in stripped:
                msg = stripped[:200]
                break

        if not msg:
            msg = exc_type

        # Determine if auto-fixable
        auto_fixable = False
        suggestion = ""

        if "MemoryError" in exc_type or "memory" in msg.lower():
            auto_fixable = True
            suggestion = "Reduce bond dimension M (halve current value, min 50)"
        elif "SyntaxError" in exc_type:
            suggestion = "Check code syntax for typos"
        elif "ImportError" in exc_type or "ModuleNotFoundError" in exc_type:
            suggestion = "Install missing package or check module path"
        elif "FileNotFoundError" in exc_type:
            suggestion = "Check input file path exists"
        elif "RuntimeError" in exc_type:
            suggestion = "Check calculation parameters and input format"
        else:
            suggestion = f"Investigate {exc_type}: {msg}"

        return ErrorInfo(
            error_type="python_error",
            message=f"{exc_type}: {msg}" if msg else exc_type,
            suggestion=suggestion,
            can_auto_fix=auto_fixable,
        )
