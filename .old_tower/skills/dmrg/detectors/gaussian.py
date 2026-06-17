"""Gaussian error detector (stub — not yet implemented).

TODO: Implement when Gaussian support is added.

Patterns to detect:
- "Error termination" in Gaussian output
- "l9999" (exceeded optimization steps)
- Link error messages ("l1", "l301", etc.)
- "Convergence failure" in SCF
- "No such file or directory" for input files
"""
from tower.graph.hooks import ErrorInfo


class GaussianErrorDetector:
    """Detect Gaussian calculation errors.

    This is a stub — implement detect() when Gaussian support is added.
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None
