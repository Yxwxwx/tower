"""Respond node hooks for DMRG skill — format results.

Extension point for domain-specific output formatting.
"""
from tower.graph.hooks import RespondHooks


class DmrgRespondHooks:
    """Respond node hooks — format computation results.

    Future formatting:
    - Energy table with unit (Hartree)
    - Convergence summary
    - Bond dimension sweep history
    """

    def format_result(self, results: dict, state: dict) -> str:
        """Format results as a structured report.

        Currently returns empty string (no extra formatting).
        Override to add domain-specific result formatting.
        """
        return ""
