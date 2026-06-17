"""VASP error detector (stub — not yet implemented).

TODO: Implement when VASP support is added.

Patterns to detect:
- "ERROR" in OUTCAR
- Electronic convergence failures ("EDDDAV", "ZPOTRF")
- Ionic convergence failures
- "VERY BAD NEWS" / "ERROR FEXCP"
- K-point errors
- Memory-related errors
"""
from tower.graph.hooks import ErrorInfo


class VaspErrorDetector:
    """Detect VASP calculation errors.

    This is a stub — implement detect() when VASP support is added.
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None
