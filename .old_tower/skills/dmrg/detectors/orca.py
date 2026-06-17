"""ORCA error detector (stub — not yet implemented).

TODO: Implement when ORCA support is added.

Patterns to detect:
- "ORCA finished with error"
- "SCF FAILED TO CONVERGE" in ORCA output
- "ORCA TERMINATED ABNORMALLY"
- "Error in geometry optimization"
- Basis set errors ("basis set not found")
"""
from tower.graph.hooks import ErrorInfo


class OrcaErrorDetector:
    """Detect ORCA calculation errors.

    This is a stub — implement detect() when ORCA support is added.
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None
