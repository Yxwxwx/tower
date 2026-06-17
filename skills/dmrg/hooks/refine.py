"""Refine node hooks for DMRG skill — fix strategies.

Maps error_type to fix strategy function.
Each strategy: (error_info: dict, step: dict) → CorrectionAction
"""
from tower.graph.hooks import CorrectionAction


class DmrgRefineHooks:
    """Fix strategies for quantum chemistry computation errors.

    Currently implemented:
    - scf_not_converged → double bond dimension M
    - python_error (MemoryError) → halve bond dimension M
    - python_error (other) → ask_user
    """

    def get_fix_strategies(self) -> dict:
        return {
            "scf_not_converged": self._fix_scf,
            "python_error": self._fix_python_error,
        }

    # ═══════════════════════════════════════════════
    # SCF Convergence
    # ═══════════════════════════════════════════════

    def _fix_scf(self, error_info: dict, step: dict) -> CorrectionAction:
        """Increase bond dimension M when SCF fails to converge.

        Doubles M up to a maximum of 2000.
        """
        current_M = self._extract_M(step, default=100)
        new_M = min(current_M * 2, 2000)

        return CorrectionAction(
            action="retry_with_params",
            new_params={**step.get("args", {}), "M": new_M},
        )

    # ═══════════════════════════════════════════════
    # Python Runtime Errors
    # ═══════════════════════════════════════════════

    def _fix_python_error(self, error_info: dict, step: dict) -> CorrectionAction:
        """Handle Python errors — only auto-fix MemoryError."""
        msg = error_info.get("message", "")

        # MemoryError: reduce M to fit in available memory
        if "MemoryError" in msg or "memory" in msg.lower():
            current_M = self._extract_M(step, default=200)
            new_M = max(current_M // 2, 50)

            return CorrectionAction(
                action="retry_with_params",
                new_params={**step.get("args", {}), "M": new_M},
            )

        # All other Python errors require human intervention
        return CorrectionAction(action="ask_user")

    # ═══════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════

    @staticmethod
    def _extract_M(step: dict, default: int = 100) -> int:
        """Extract bond dimension M from step args, with type coercion."""
        args = step.get("args", {})
        M = args.get("M", default)
        if isinstance(M, str):
            try:
                M = int(M)
            except ValueError:
                M = default
        return M
