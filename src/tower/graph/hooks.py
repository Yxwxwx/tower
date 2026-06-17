"""Tower graph hook protocols — skill packs implement these to inject domain logic.

Each node has a corresponding hook protocol. Skills optionally implement them.
Unimplemented hooks → node falls back to default behavior.
"""
from dataclasses import dataclass
from typing import Protocol


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ErrorInfo:
    """Structured error detected by a domain error detector.

    Attributes:
        error_type: Machine-readable error category.
            e.g. "scf_not_converged", "python_error", "mpi_error"
        message: Human-readable description of what went wrong.
        suggestion: Suggested fix for the user or refine node.
        can_auto_fix: True if refine node can auto-correct and retry.
    """
    error_type: str
    message: str
    suggestion: str = ""
    can_auto_fix: bool = False


@dataclass
class CorrectionAction:
    """Corrective action produced by a refine fix strategy.

    Attributes:
        action: "retry_with_params" | "skip" | "ask_user"
        new_params: Corrected parameters (for retry_with_params).
    """
    action: str
    new_params: dict | None = None


# ═══════════════════════════════════════════════════════════════════
# Error Detector Protocol
# ═══════════════════════════════════════════════════════════════════


class ErrorDetector(Protocol):
    """Detect a specific class of computation errors.

    Each software (PySCF, Gaussian, ORCA, ...) and error type
    (SCF, MPI, OOM, ...) gets its own detector class. Observe node
    runs all registered detectors against the computation output.
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        """Check computation output for a known error pattern.

        Returns:
            ErrorInfo if the pattern is detected, None otherwise.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# Per-Node Hook Protocols
# ═══════════════════════════════════════════════════════════════════


class PlanHooks(Protocol):
    """Hooks injected into the plan node."""

    def preprocess(self, task: str, state: dict) -> str:
        """Preprocess the user's task before LLM planning.

        Can enrich the task with domain context, translate shorthand
        notation, or add implicit requirements.
        """
        ...

    def validate_plan(self, plan: list[dict]) -> list[dict]:
        """Validate and potentially modify the generated plan.

        Can reorder steps, add missing validation steps, or reject
        invalid parameter combinations.
        """
        ...


class ActHooks(Protocol):
    """Hooks injected into the act node."""

    def pre_run(self, tool_call: dict, state: dict) -> dict:
        """Validate/transform tool parameters before execution.

        Can check parameter ranges, add defaults, or convert units.
        Returns the (potentially modified) tool_call dict.
        """
        ...

    def post_run(self, result: dict, state: dict) -> dict:
        """Enrich tool result after successful execution.

        Can parse raw output, extract key metrics, or add metadata.
        Returns the enriched result dict.
        """
        ...


class ObserveHooks(Protocol):
    """Hooks injected into the observe node."""

    def get_error_detectors(self) -> list:
        """Return the list of error detectors to run.

        Detectors are run in order; the first match wins.
        Order matters: put more specific detectors first.

        Returns:
            List of objects implementing ErrorDetector protocol.
        """
        ...


class RefineHooks(Protocol):
    """Hooks injected into the refine node."""

    def get_fix_strategies(self) -> dict:
        """Return a mapping from error_type to fix strategy function.

        Each strategy: (error_info: dict, step: dict) → CorrectionAction

        Example:
            {"scf_not_converged": self._fix_scf_convergence}
        """
        ...


class RespondHooks(Protocol):
    """Hooks injected into the respond node."""

    def format_result(self, results: dict, state: dict) -> str:
        """Format computation results for user-facing output.

        Can extract key metrics, format tables, add units.
        Returns a formatted string to prepend to the LLM response.
        """
        ...
