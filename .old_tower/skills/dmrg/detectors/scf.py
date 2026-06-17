"""SCF convergence error detector for PySCF.

Detects SCF convergence failures in computation output.
"""
import re
from tower.graph.hooks import ErrorInfo


class ScfConvergenceDetector:
    """Detect SCF convergence failures in PySCF output.

    Matches patterns:
    - "SCF not converged" (PySCF)
    - "SCF FAILED TO CONVERGE" (ORCA — detection only, auto-fix not implemented)
    """

    def detect(self, stdout: str, stderr: str, exit_code: int) -> ErrorInfo | None:
        combined = stdout + "\n" + stderr

        # PySCF pattern
        if "SCF not converged" in combined or "scf not converged" in combined.lower():
            niter = "?"
            m = re.search(r"niter\s*[=:]\s*(\d+)", combined)
            if m:
                niter = m.group(1)

            # Try to extract current M value for a better suggestion
            M = None
            m_m = re.search(r"[Mm]\s*[=:]\s*(\d+)", combined)
            if m_m:
                M = int(m_m.group(1))

            suggestion = "Increase bond dimension M"
            if M:
                suggestion += f" (current M={M}, try M={min(M * 2, 2000)})"

            return ErrorInfo(
                error_type="scf_not_converged",
                message=f"SCF 未收敛 (niter={niter})",
                suggestion=suggestion,
                can_auto_fix=True,
            )

        # ORCA pattern (stub — detection only, no auto-fix yet)
        if "SCF FAILED TO CONVERGE" in combined:
            return ErrorInfo(
                error_type="scf_not_converged",
                message="ORCA SCF 收敛失败",
                suggestion="尝试 'slowconv' 关键词或降低收敛阈值",
                can_auto_fix=False,
            )

        return None
