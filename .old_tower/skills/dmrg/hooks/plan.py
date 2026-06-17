"""Plan node hooks for DMRG skill.

Extension point for pre-processing tasks and validating plans.
"""
from tower.graph.hooks import PlanHooks


class DmrgPlanHooks:
    """Plan node hooks — task preprocessing and plan validation.

    Currently minimal. Extend as domain logic is added.
    """

    def preprocess(self, task: str, state: dict) -> str:
        """Enrich task with domain context if needed.

        Future: detect shorthand notation (e.g. 'H4 M=200') and expand
        to full task description with default parameters.
        """
        return task

    def validate_plan(self, plan: list[dict]) -> list[dict]:
        """Validate the generated calculation plan.

        Future checks:
        - Ensure bond dimension M is within reasonable range (50-2000)
        - Ensure required input files are referenced
        - Add convergence-check step after each computation step
        """
        return plan
