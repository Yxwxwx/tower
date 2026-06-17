"""Tower CLI — entry point for the multi-agent system."""
import asyncio
import uuid

import click
from rich.console import Console
from rich.panel import Panel

from tower.state.run_store import RunState, RunStateStore

console = Console()


@click.group()
def cli():
    """Tower — supervisor-led modular multi-agent system for computational chemistry."""
    pass


@cli.command()
@click.argument("task")
@click.option("--run-id", "-r", default=None, help="Run ID (auto-generated if not set)")
def run(task: str, run_id: str | None):
    """Execute a computational chemistry task.

    Example: tower run "N2 NEVPT2 calculation"
    """
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4().hex[:8]}"

    console.print(Panel.fit(
        f"[bold]Task:[/] {task}\n"
        f"[dim]Run ID: {run_id}[/]\n"
        f"[dim]Trace ID: {trace_id}[/]",
        title="[bold blue]Tower Multi-Agent System[/]",
        border_style="blue",
    ))

    async def _run():
        store = RunStateStore()
        run_state = RunState(
            run_id=run_id,
            trace_id=trace_id,
            task=task,
        )
        await store.create(run_state)

        # Load agents
        from agents.supervisor import register as sv_reg
        from agents.gaussian import register as ga_reg
        from agents.pyscf import register as py_reg
        from agents.orca import register as or_reg
        from agents.hpc import register as hp_reg
        from agents.monitor import register as mo_reg

        agents = {
            r.name: r for r in [
                sv_reg(), ga_reg(), py_reg(), or_reg(), hp_reg(), mo_reg(),
            ]
        }

        console.print(f"[dim]Loaded {len(agents)} agents:[/] "
                      f"{', '.join(agents.keys())}")

        # Invoke supervisor
        supervisor = agents["supervisor"]
        initial_state = {
            "task": task,
            "run_id": run_id,
            "trace_id": trace_id,
        }
        result = supervisor.subgraph.invoke(initial_state)

        console.print()
        console.print(Panel(
            result.get("final_response", "No response"),
            title="[bold green]Result[/]",
            border_style="green",
        ))
        console.print(f"[dim]Plan: {' → '.join(result.get('plan', []))}[/]")

    asyncio.run(_run())


@cli.command()
def agents():
    """List all registered agents."""
    from agents.supervisor import register as sv_reg
    from agents.gaussian import register as ga_reg
    from agents.pyscf import register as py_reg
    from agents.orca import register as or_reg
    from agents.hpc import register as hp_reg
    from agents.monitor import register as mo_reg

    registrations = [
        sv_reg(), ga_reg(), py_reg(), or_reg(), hp_reg(), mo_reg(),
    ]

    for r in registrations:
        deps = ", ".join(sorted(r.dependencies)) if r.dependencies else "none"
        console.print(
            f"[bold]{r.name}[/] "
            f"[dim]retry={r.retry_policy.max_retries}x "
            f"timeout={r.timeout_s}s "
            f"deps=[{deps}][/]"
        )
        console.print(f"  {r.description}")


def main():
    cli()


if __name__ == "__main__":
    main()
