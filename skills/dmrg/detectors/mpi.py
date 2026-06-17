"""MPI error detector (stub — not yet implemented).

TODO: Implement MPI error pattern matching for common MPI implementations
(OpenMPI, MPICH, Intel MPI).

Patterns to detect:
- "MPI_Abort"
- "MPI Error"
- Rank-specific error messages
- "ORTE_ERROR_LOG"
"""
from tower.graph.hooks import ErrorInfo


class MpiErrorDetector:
    """Detect MPI errors in parallel computation output.

    This is a stub — it returns None for all inputs until
    MPI error patterns are implemented.
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        return None
