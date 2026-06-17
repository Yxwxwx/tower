"""Tower CLI — supervisor-led modular multi-agent system for computational chemistry.

Rendering stack: Click (CLI framework) + Rich (terminal rendering).
"""
import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Ensure agents/ is importable (project root on sys.path)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.rule import Rule
from rich.columns import Columns
from rich import box

from tower.state.run_store import RunState, RunStateStore

console = Console()

# ═══════════════════════════════════════════════════════════════════
# Branding
# ═══════════════════════════════════════════════════════════════════

TOWER_BANNER = r"""
[bold blue]   ████████╗ ██████╗ ██╗    ██╗███████╗██████╗
   ╚══██╔══╝██╔═══██╗██║    ██║██╔════╝██╔══██╗
      ██║   ██║   ██║██║ █╗ ██║█████╗  ██████╔╝
      ██║   ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
      ██║   ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
      ╚═╝    ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝[/]
[dim]      Supervisor-led Modular Multi-Agent System for Computational Chemistry[/]

[dim]   version 0.2.0  ·  LangGraph + MCP  ·  contracts v1[/]
"""


def _print_banner():
    console.print(TOWER_BANNER)


# ═══════════════════════════════════════════════════════════════════
# Status helpers
# ═══════════════════════════════════════════════════════════════════

STATUS_COLORS = {
    "pending":   "dim",
    "running":   "bold yellow",
    "done":      "bold green",
    "failed":    "bold red",
    "retrying":  "bold magenta",
    "abandoned": "red",
    "needs_human": "bold yellow",
}

STATUS_ICONS = {
    "pending":   "○",
    "running":   "◌",
    "done":      "●",
    "failed":    "✕",
    "retrying":  "↺",
    "abandoned": "✕",
    "needs_human": "?",  # filled by _needs_human logic
}


def _status_label(status: str) -> Text:
    """Colored status label with icon."""
    color = STATUS_COLORS.get(status, "white")
    icon = STATUS_ICONS.get(status, "·")
    return Text(f"{icon} {status}", style=color)


def _agent_name_cell(name: str) -> Text:
    """Colored agent name based on type."""
    if name == "supervisor":
        return Text(name, style="bold cyan")
    elif name in ("hpc", "monitor"):
        return Text(name, style="bold magenta")
    else:
        return Text(name, style="bold green")


# ═══════════════════════════════════════════════════════════════════
# Agent listing (tower agents)
# ═══════════════════════════════════════════════════════════════════

def _render_agent_list(registrations: list) -> Table:
    """Render registered agents as a Rich Table."""
    table = Table(
        title="Registered Agents",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold white",
        show_lines=False,
    )
    table.add_column("Agent", style="bold", width=14)
    table.add_column("Type", width=14)
    table.add_column("Retries", justify="center", width=8)
    table.add_column("Timeout", justify="right", width=10)
    table.add_column("Description", width=66)

    for r in sorted(registrations, key=lambda x: (
        0 if x.name == "supervisor" else
        1 if x.name in ("gaussian", "pyscf", "orca") else 2,
        x.name,
    )):
        if r.name in ("hpc", "monitor"):
            agent_type = "[magenta]infra[/]"
        elif r.name == "supervisor":
            agent_type = "[cyan]orchestrator[/]"
        else:
            agent_type = "[green]domain[/]"

        timeout = f"{r.timeout_s // 3600}h" if r.timeout_s >= 3600 else f"{r.timeout_s}s"

        table.add_row(
            _agent_name_cell(r.name),
            agent_type,
            str(r.retry_policy.max_retries),
            timeout,
            r.description[:100],
        )

    return table


# ═══════════════════════════════════════════════════════════════════
# Run execution rendering
# ═══════════════════════════════════════════════════════════════════

def _render_run_header(run_id: str, trace_id: str, task: str):
    """Render the run start panel."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=12)
    grid.add_column()
    grid.add_row("Task", f"[bold white]{task}[/]")
    grid.add_row("Run ID", f"[dim]{run_id}[/]")
    grid.add_row("Trace ID", f"[dim]{trace_id}[/]")
    grid.add_row("Time", f"[dim]{time.strftime('%Y-%m-%d %H:%M:%S')}[/]")

    console.print()
    console.print(Panel(
        grid,
        title="[bold blue]⚛ Tower Run[/]",
        border_style="blue",
        padding=(1, 2),
    ))


def _render_plan(plan: list[str]):
    """Render the supervisor's execution plan."""
    if not plan:
        return
    parts = []
    for i, agent in enumerate(plan):
        color = "cyan" if agent == "supervisor" else "green"
        parts.append(Text(f"[{color}]{agent}[/]", style=color))
        if i < len(plan) - 1:
            parts.append(Text(" → ", style="dim"))

    console.print()
    console.print(
        Panel(Text.assemble(*parts),
              title="[bold]Plan[/]", border_style="green", padding=(1, 2)))


def _render_agent_result(agent_name: str, status: str, data=None, errors=None):
    """Render a single agent's execution result."""
    color = STATUS_COLORS.get(status, "white")
    icon = "✓" if status == "done" else "✕" if status == "failed" else "·"

    lines = [f"[{color}]{icon} {status.upper()}[/] "]

    if data and hasattr(data, "energy") and data.energy is not None:
        lines.append(f"[dim]E = {data.energy:.6f} Ha[/]")
    if data and hasattr(data, "converged") and data.converged:
        lines.append("[green]converged[/]")
    if data and hasattr(data, "wall_time_s") and data.wall_time_s > 0:
        h = data.wall_time_s / 3600
        lines.append(f"[dim]{h:.1f}h[/]")

    if errors:
        for e in errors[:3]:
            lines.append(f"[red]{e[:80]}[/]")

    console.print(f"  {_agent_name_cell(agent_name)}  {' '.join(lines)}")


def _render_final_result(final_response: str, plan: list[str], node_history: list[str] = None):
    """Render the final result panel."""
    console.print()
    console.print(Rule(style="green"))

    if final_response:
        console.print(Panel(
            final_response,
            title="[bold green]Result[/]",
            border_style="green",
            padding=(1, 2),
        ))

    # Footer
    footer = Text()
    footer.append(f"plan: {' → '.join(plan)}", style="dim")
    if node_history:
        footer.append(f"  ·  nodes: {' → '.join(node_history)}", style="dim")

    console.print(footer)
    console.print(Rule(style="green"))


# ═══════════════════════════════════════════════════════════════════
# CLI commands
# ═══════════════════════════════════════════════════════════════════

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Tower — supervisor-led multi-agent system for computational chemistry."""
    if ctx.invoked_subcommand is None:
        _print_banner()
        console.print()
        console.print("[dim]Commands:[/]")
        console.print("  [bold]tower run[/] <task>     Execute a computational chemistry task")
        console.print("  [bold]tower agents[/]           List all registered agents")
        console.print()
        console.print("[dim]Examples:[/]")
        console.print("  [dim]$[/] tower run \"N2 NEVPT2 calculation\"")
        console.print("  [dim]$[/] tower run \"H4 chain DMRG with M=200\"")
        console.print("  [dim]$[/] tower agents")


@cli.command()
@click.argument("task")
@click.option("--run-id", "-r", default=None, help="Run ID (auto-generated)")
def run(task: str, run_id: Optional[str]):
    """Execute a computational chemistry task."""
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4().hex[:8]}"

    _print_banner()
    _render_run_header(run_id, trace_id, task)

    async def _run():
        store = RunStateStore()
        run_state = RunState(run_id=run_id, trace_id=trace_id, task=task)
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

        # Show loaded agents
        agent_list = ", ".join(
            str(_agent_name_cell(n)) for n in agents
        )
        console.print(f"  [dim]agents:[/] {agent_list}")

        # Invoke supervisor
        supervisor = agents["supervisor"]
        initial_state = {
            "task": task,
            "run_id": run_id,
            "trace_id": trace_id,
        }

        with console.status("[bold green]supervisor planning...[/]", spinner="dots"):
            result = supervisor.subgraph.invoke(initial_state)

        plan = result.get("plan", [])
        _render_plan(plan)

        # Execute each agent in the plan (MVP: simulate with status)
        for i, agent_name in enumerate(plan):
            if agent_name not in agents or agent_name == "supervisor":
                continue

            agent_reg = agents[agent_name]
            step_label = f"[bold]{agent_name}[/] ({i+1}/{len(plan)})"

            with console.status(f"  {step_label} [dim]executing...[/]", spinner="dots"):
                time.sleep(0.3)  # MVP: simulate execution
                # In production: agent_reg.subgraph.invoke(state)

            _render_agent_result(agent_name, "done")

        # Final result
        _render_final_result(result.get("final_response", ""), plan)

    asyncio.run(_run())


@cli.command()
def agents():
    """List all registered agents and their contracts."""
    _print_banner()

    from agents.supervisor import register as sv_reg
    from agents.gaussian import register as ga_reg
    from agents.pyscf import register as py_reg
    from agents.orca import register as or_reg
    from agents.hpc import register as hp_reg
    from agents.monitor import register as mo_reg

    registrations = [
        sv_reg(), ga_reg(), py_reg(), or_reg(), hp_reg(), mo_reg(),
    ]

    console.print()
    console.print(_render_agent_list(registrations))
    console.print()

    # Architecture note
    console.print()
    console.print(Panel(
        "[dim]All agents are independent. The [cyan]supervisor[/] alone decides "
        "execution order and data flow. Agents have no knowledge of each other — "
        "the supervisor passes artifacts between them via [bold]artifacts_in[/].[/]",
        title="[bold]Architecture[/]",
        border_style="dim blue",
        padding=(1, 2),
    ))


def main():
    cli()


if __name__ == "__main__":
    main()
