"""Tower Chat — 交互式对话 CLI，类似 Claude Code 的 REPL 体验。

特性:
- Rich Markdown 渲染 (代码高亮, 列表, 表格)
- DeepSeek thinking 折叠 (Ctrl+O 展开)
- 分隔线区分每轮对话
- 工具调用可视化
- 多轮对话上下文管理
"""

import asyncio
import json
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from tower.llm import get_model
from tower.tools.builtin.filesystem import (
    read_file,
    write_file,
    edit_file,
    list_directory,
    glob_tool,
    grep,
    move_file,
    copy_file,
    delete_file,
)
from tower.tools.builtin.bash import bash
from tower.tools.builtin.web import web_fetch, web_search

console = Console()

# ═══════════════════════════════════════════════════════════════════
# Visual constants
# ═══════════════════════════════════════════════════════════════════

SEP = Rule(style="dim", characters="─")

# ═══════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are Tower, a general-purpose AI assistant powered by DeepSeek.

## Capabilities
- Read, write, edit, search, and manage files
- Execute bash commands
- Fetch web pages and search the web
- Analyze code, debug issues, explain concepts
- Generate scripts, configs, and documentation
- Dispatch computational chemistry tasks to a multi-agent pipeline

## Guidelines
- Be concise and accurate. Use tools when they help answer the question.
- When editing files, use edit_file (not write_file) for targeted changes — it's safer.
- Read files before editing them to understand the current content.
- When searching code, prefer grep for content searches and glob_tool for file name patterns.
- Explain what you're doing before taking destructive actions (delete_file, bash rm).
- Use Markdown for code blocks, lists, and tables in your responses.
- If you're unsure, ask clarifying questions rather than guessing.

## Computational Chemistry — IMPORTANT
Quantum chemistry calculations (DFT, HF, CASSCF, NEVPT2, CCSD, geometry
optimizations, frequency analysis, etc.) MUST go through the multi-agent
pipeline, not run directly via bash.

**DO NOT:**
- pip install pyscf / conda install quantum chemistry packages
- Run python scripts that import pyscf, gaussian, orca, etc. directly
- Generate and execute computational chemistry scripts yourself

**ALWAYS do this instead:**
Suggest the user run: /run <their task description>
Example: /run 用pyscf计算N2分子的PBE/cc-pVDZ能量

The /run command dispatches to specialized agents (PySCF, Gaussian, Orca,
HPC, Monitor) that handle input generation, HPC job submission, monitoring,
and output parsing. You are the chat assistant — leave the computation to
the agents.

## Tools at your disposal
You have access to: read_file, write_file, edit_file, list_directory, glob_tool,
grep, move_file, copy_file, delete_file, bash, web_fetch, web_search.

Use them when appropriate — don't describe what you would do, actually do it."""

# ═══════════════════════════════════════════════════════════════════
# Available Tools
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    read_file,
    write_file,
    edit_file,
    list_directory,
    glob_tool,
    grep,
    move_file,
    copy_file,
    delete_file,
    bash,
    web_fetch,
    web_search,
]

TOOL_BY_NAME = {t.name: t for t in TOOLS}

# Tools that modify the system — require user confirmation
_DANGEROUS_TOOLS = {"bash", "write_file", "edit_file", "delete_file", "move_file", "copy_file"}

# ═══════════════════════════════════════════════════════════════════
# Thinking storage — for Ctrl+O reveal
# ═══════════════════════════════════════════════════════════════════

# Mutable container: stores the most recent thinking content.
# Ctrl+O prints it when pressed.
_last_thinking = {"content": ""}


# ═══════════════════════════════════════════════════════════════════
# Prompt-toolkit session
# ═══════════════════════════════════════════════════════════════════

_history_path = Path.home() / ".tower_chat_history"
PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "bold green",
        "input": "",
    }
)


def _get_session() -> PromptSession:
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("c-o")
    def _(event):
        """Open thinking content in pager (new page)."""
        content = _last_thinking.get("content", "")
        if not content:
            console.print("\n[dim]No thinking content to show.[/]")
            return

        # Open a new "page" — uses system pager (less/more)
        panel = Panel(
            content,
            border_style="dim blue",
            padding=(1, 2),
            title="[bold blue]Thinking[/]",
            title_align="left",
            subtitle="[dim]q to close, arrows to scroll[/]",
            subtitle_align="right",
        )
        with console.pager(styles=True):
            console.print(panel)

    return PromptSession(
        history=FileHistory(str(_history_path)),
        style=PROMPT_STYLE,
        message=[
            ("class:prompt", "\n▸ "),
        ],
        key_bindings=kb,
    )


# ═══════════════════════════════════════════════════════════════════
# Tool execution & rendering
# ═══════════════════════════════════════════════════════════════════


def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool by name and return the result as a JSON string."""
    tool_fn = TOOL_BY_NAME.get(tool_name)
    if tool_fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = tool_fn.invoke(tool_args)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _ask_approval(tool_name: str, args: dict) -> tuple[bool, bool]:
    """Ask user to approve a dangerous tool call.

    Returns:
        (approved: bool, skip_remaining: bool)
    """
    # Compact description of what's about to happen
    if tool_name == "bash":
        desc = args.get("command", "?")[:100]
    elif tool_name in ("write_file", "edit_file"):
        desc = args.get("path", "?")
    elif tool_name == "delete_file":
        desc = f"DELETE {args.get('path', '?')}"
    elif tool_name in ("move_file", "copy_file"):
        desc = f"{args.get('source', '?')} → {args.get('destination', '?')}"
    else:
        desc = json.dumps(args, ensure_ascii=False)[:100]

    console.print(
        f"  [bold yellow]⚠ {tool_name}[/] [dim]{desc}[/]"
    )
    console.print(
        "  [dim][y] yes  [n] no  [a] yes to all[/] ",
        end="",
    )

    try:
        choice = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False, False

    if choice == "a":
        return True, True  # approved + skip remaining
    if choice in ("y", "yes", ""):
        return True, False
    return False, False


def _render_tool_call(tool_name: str, args: dict):
    """Render a tool call — compact, like Claude Code."""
    display_args = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 80:
            display_args[k] = v[:80] + "..."
        else:
            display_args[k] = v

    console.print(
        Panel(
            f"[bold cyan]{tool_name}[/] [dim]{json.dumps(display_args, ensure_ascii=False)}[/]",
            border_style="cyan",
            padding=(0, 1),
            title="[bold cyan]⚙ TOOL[/]",
            title_align="left",
        )
    )


def _render_tool_result(result: str):
    """Render a tool result — compact."""
    display = result[:2000]
    if len(result) > 2000:
        display += f"\n[dim]... ({len(result) - 2000} more chars)[/]"

    console.print(
        Panel(
            display,
            border_style="dim cyan",
            padding=(0, 1),
            title="[dim]↳ RESULT[/]",
            title_align="left",
        )
    )


# ═══════════════════════════════════════════════════════════════════
# Response streaming with thinking animation
# ═══════════════════════════════════════════════════════════════════


async def _invoke_with_thinking(model, messages: list) -> tuple[AIMessage, str]:
    """Invoke the model with a thinking spinner, then render the response.

    Thinking content is COLLAPSED by default — not displayed.
    Use Ctrl+O to reveal it.

    Returns:
        (merged AIMessage, thinking_content_string)
    """
    collected_content = ""
    collected_chunks = []
    reasoning_snippets: list[str] = []
    has_reasoning = False

    spinner = Spinner("dots", text="[bold blue]Thinking[/] [dim]...[/]")

    with Live(spinner, refresh_per_second=8, transient=True) as live:
        async for chunk in model.astream(messages):
            collected_chunks.append(chunk)

            # Track reasoning (collapsed by default — Ctrl+O to reveal)
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                has_reasoning = True
                reasoning_snippets.append(reasoning)

            # Actual content arriving
            if chunk.content:
                collected_content += chunk.content
                live.update(
                    Spinner("dots", text="[bold blue]Generating[/] [dim]...[/]")
                )

    # Merge chunks
    if collected_chunks:
        full = collected_chunks[0]
        for c in collected_chunks[1:]:
            full += c
    else:
        full = AIMessage(content=collected_content)

    thinking_text = "".join(reasoning_snippets).strip()

    # Render content as Markdown
    if collected_content.strip():
        console.print(SEP)
        console.print()
        md = Markdown(
            collected_content, code_theme="monokai", inline_code_theme="monokai"
        )
        console.print(md)
        console.print()

    # Collapsed thinking indicator (if thinking exists)
    if thinking_text:
        size_k = len(thinking_text) / 1000
        console.print(
            f"[dim]💭 Thinking · {size_k:.1f}k chars (Ctrl+O to expand)[/]"
        )

    return full, thinking_text


# ═══════════════════════════════════════════════════════════════════
# Multi-agent dispatch (/run)
# ═══════════════════════════════════════════════════════════════════


async def _dispatch_run(task: str):
    """Invoke the multi-agent supervisor pipeline for a computational task."""
    import uuid

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4().hex[:8]}"

    # Load all agents
    from agents.supervisor import register as sv_reg
    from agents.gaussian import register as ga_reg
    from agents.pyscf import register as py_reg
    from agents.orca import register as or_reg
    from agents.hpc import register as hp_reg
    from agents.monitor import register as mo_reg

    registrations = [
        sv_reg(), ga_reg(), py_reg(), or_reg(), hp_reg(), mo_reg(),
    ]
    agents = {r.name: r for r in registrations}

    # Wire agent subgraphs into supervisor's dispatch node
    from agents.supervisor.agent import set_agent_registry
    subgraphs = {r.name: r.subgraph for r in registrations if r.name != "supervisor"}
    set_agent_registry(subgraphs)

    # Header
    console.print(SEP)
    console.print(
        Panel(
            f"[bold white]{task}[/]\n"
            f"[dim]run: {run_id}  ·  trace: {trace_id}  ·  {len(agents)} agents[/]",
            title="[bold blue]⚛ Dispatch[/]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Invoke supervisor
    supervisor = agents["supervisor"]
    initial_state = {
        "task": task,
        "run_id": run_id,
        "trace_id": trace_id,
    }

    with console.status("[bold green]supervisor orchestrating...[/]", spinner="dots"):
        result = supervisor.subgraph.invoke(initial_state)

    plan = result.get("plan", [])

    # Render plan
    parts = []
    for i, name in enumerate(plan):
        color = "cyan" if name == "supervisor" else "green"
        parts.append(Text(name, style=f"bold {color}"))
        if i < len(plan) - 1:
            parts.append(Text(" → ", style="dim"))
    console.print(Panel(Text.assemble(*parts), title="[bold]Plan[/]", border_style="green", padding=(1, 2)))

    # Render per-agent results
    agent_results = result.get("agent_results", {})
    for name in plan:
        ar = agent_results.get(name)
        if ar is None:
            console.print(f"  [dim]○ {name}: pending[/]")
            continue
        status = ar.status.value if hasattr(ar, "status") else "?"
        icon = "✓" if status == "done" else "✕" if status == "failed" else "○"
        color = {"done": "green", "failed": "red"}.get(status, "dim")
        line = f"  [{color}]{icon} {name}: {status}[/]"

        if hasattr(ar, "data") and ar.data:
            ds = str(ar.data)[:120]
            line += f" [dim]{ds}[/]"
        if hasattr(ar, "errors") and ar.errors:
            for e in ar.errors[:2]:
                line += f"\n    [red]{e[:100]}[/]"
        console.print(line)

    # Final result
    final = result.get("final_response", "")
    if final:
        console.print(SEP)
        console.print(Markdown(final))
        console.print()

    console.print(SEP)


# ═══════════════════════════════════════════════════════════════════
# Main chat loop
# ═══════════════════════════════════════════════════════════════════


async def run_chat():
    """运行交互式对话循环。"""
    console.clear()
    console.print()
    console.print(
        Panel(
            "[bold blue]Tower Chat[/] — interactive AI assistant\n"
            "[dim]DeepSeek-powered · tools enabled · Markdown rendering[/]\n\n"
            "  Type your message, or:\n"
            "  [bold]/exit[/]   [dim]— quit[/]\n"
            "  [bold]/clear[/]  [dim]— reset conversation[/]\n"
            "  [bold]/run[/]    [dim]— dispatch to multi-agent system[/]\n"
            "  [bold]/help[/]   [dim]— show this message[/]",
            border_style="blue",
            padding=(1, 2),
            title="[bold]⚛ Tower[/]",
            title_align="left",
        )
    )

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    session = _get_session()

    while True:
        try:
            user_input = await session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            return

        user_input = user_input.strip()
        if not user_input:
            continue

        # ── Built-in commands (/ prefix only; \ is for LaTeX) ──
        if user_input.startswith("/"):
            cmd = user_input.lstrip("/").strip().lower()

            if cmd in ("exit", "quit", "q"):
                console.print("[dim]Goodbye.[/]")
                return
            if cmd.startswith("run"):
                task = user_input[4:].strip()  # everything after "/run"
                if not task:
                    console.print("[dim]Usage: /run <task> — e.g. /run N2 NEVPT2[/]")
                    continue
                await _dispatch_run(task)
                continue
            if cmd in ("clear", "cls"):
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                _last_thinking["content"] = ""
                console.clear()
                console.print("[dim]Conversation reset.[/]\n")
                continue
            if cmd in ("help", "h", "?"):
                console.print(
                    Panel(
                        "[bold]/exit[/]   [dim]Quit Tower Chat[/]\n"
                        "[bold]/clear[/]  [dim]Reset conversation history[/]\n"
                        "[bold]Ctrl+O[/]   [dim]View thinking content[/]\n"
                        "[bold]/help[/]   [dim]Show this message[/]",
                        border_style="dim blue",
                        padding=(1, 2),
                    )
                )
                continue

            # Unknown command — don't send to LLM
            console.print(
                f"[dim]Unknown command: {user_input} — try /help[/]"
            )
            continue

        # ── Send to LLM ──
        messages.append(HumanMessage(content=user_input))

        model = get_model().bind_tools(TOOLS)

        response, thinking = await _invoke_with_thinking(model, messages)

        # Store thinking for Ctrl+O reveal
        _last_thinking["content"] = thinking

        # ── Handle tool calls ──
        skip_confirm = False  # reset per user message

        while response.tool_calls:
            tool_msgs = []

            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", "")

                _render_tool_call(tool_name, tool_args)

                # ── User confirmation for dangerous tools ──
                if tool_name in _DANGEROUS_TOOLS and not skip_confirm:
                    approved, skip_confirm = _ask_approval(tool_name, tool_args)
                    if not approved:
                        result = json.dumps(
                            {"error": "User denied this operation.", "denied": True},
                            ensure_ascii=False,
                        )
                        _render_tool_result(result)
                        tool_msgs.append(
                            ToolMessage(content=result, tool_call_id=tool_call_id)
                        )
                        continue

                result = _execute_tool(tool_name, tool_args)
                _render_tool_result(result)

                tool_msgs.append(ToolMessage(content=result, tool_call_id=tool_call_id))

            messages.append(response)
            messages.extend(tool_msgs)

            # Follow-up after tool use
            response, thinking = await _invoke_with_thinking(model, messages)

            # Update thinking for Ctrl+O (last thinking wins)
            if thinking:
                _last_thinking["content"] = thinking

        messages.append(response)
