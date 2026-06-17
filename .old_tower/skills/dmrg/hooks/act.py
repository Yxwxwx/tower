"""Act node hooks for DMRG skill.

Validate parameters before computation, enrich results after.
"""
from tower.graph.hooks import ActHooks


class DmrgActHooks:
    """Act node hooks — parameter validation and result enrichment.

    pre_run: validate DMRG parameters before execution
    post_run: parse energy values and convergence status from output
    """

    def pre_run(self, tool_call: dict, state: dict) -> dict:
        """Validate DMRG parameters before running.

        Future checks:
        - M range: 50 ≤ M ≤ 2000
        - Required params: model, M present
        - Input file exists
        """
        return tool_call.get("args", {})

    def post_run(self, result: dict, state: dict) -> dict:
        """Enrich computation result with parsed metrics.

        Future parsing:
        - Extract ground state energy (in Hartree)
        - Extract bond dimension used
        - Extract sweep count and final truncation error
        - Extract SCF convergence status
        """
        return result
