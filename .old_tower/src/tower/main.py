"""Tower CLI — Plan→Act→Observe→Refine→Respond agent。"""
import asyncio
import os
import uuid
import traceback
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from tower.runtime.orchestrator import build_graph
from tower.runtime.skill_loader import SkillLoader
from tower.graph.nodes import _get_llm
from tower.memory import create_checkpointer, LongTermMemory, close_pool

load_dotenv()
console = Console()

DB_URI = os.getenv("TOWER_DB", "postgresql://sunxinyu@localhost:5432/tower")


def _is_casual_chat(text: str) -> bool:
    """快速判断用户输入是否为闲聊，避免不必要的 LLM 调用。"""
    t = text.strip().lower()
    if len(t) < 10:
        return True
    casual = {"hello", "hi", "hey", "thanks", "thank you", "ok", "okay",
              "good", "great", "bye", "goodbye", "yes", "no", "what",
              "help", "test", "testing", "nice", "cool", "thanks!",
              "what time is it", "how are you", "good morning",
              "good afternoon", "good evening", "good night"}
    if t in casual:
        return True
    return False


async def extract_memories(task: str, response: str, llm, ltm: LongTermMemory):
    """分析对话，提取关于用户的长期记忆事实。同时处理矛盾信息。"""
    if _is_casual_chat(task):
        return

    # 获取已有事实，帮助 LLM 识别矛盾
    try:
        existing_facts = await ltm.get_all_facts()
        existing_text = "\n".join(f"- [{f.get('category', '')}] {f['fact']}" for f in existing_facts)
    except Exception:
        existing_text = "(无法获取已有记忆)"

    try:
        result = llm.invoke([HumanMessage(content=(
            f"分析以下对话，提取关于用户的、值得跨会话记住的信息。\n"
            f"规则：\n"
            f"1. 尽量使用用户的原始措辞，不要缩写专业术语\n"
            f"2. 每条事实独立，不要合并\n"
            f"3. 如果没有值得记住的新信息，回复 NO_FACTS\n"
            f"4. 如果用户的新说法与已有记忆矛盾，在矛盾的事实前加 REMOVE: 前缀\n"
            f"5. 格式：每条事实一行\n\n"
            f"已有记忆：\n{existing_text}\n\n"
            f"用户消息: {task}\n"
            f"助手回复: {response[:600]}"
        ))])
    except Exception:
        return

    text = result.content.strip() if hasattr(result, "content") else ""

    if "NO_FACTS" in text.upper():
        return

    memories = []
    for line in text.split("\n"):
        line = line.strip().lstrip("- 0123456789. ")
        if not line or len(line) <= 5:
            continue

        # 处理矛盾移除
        if line.upper().startswith("REMOVE:"):
            old_fact = line[7:].strip()
            try:
                await ltm.remove_fact_by_text(old_fact)
                console.print(Text(f"  🗑️ forgot: {old_fact}", style="dim"))
            except Exception:
                pass
            continue

        try:
            await ltm.add_fact(line, category="user_profile")
            memories.append(line)
        except Exception:
            pass

    if memories:
        console.print(Text("  🧠 remembered:", style="dim"))
        for m in memories:
            console.print(Text(f"     {m}", style="dim italic"))


class TowerChat:
    """Tower 对话会话。"""

    def __init__(self, thread_id: str | None = None, user_id: str = "default",
                 skill_path: str | None = None):
        import threading
        self.user_id = user_id
        self.thread_id = thread_id or f"session-{uuid.uuid4().hex[:8]}"
        self.config = {"configurable": {"thread_id": self.thread_id}}

        # Skill pack（领域知识注入）
        self.skill = None
        if skill_path:
            self.skill = SkillLoader.load(skill_path)
            if self.skill:
                console.print(Text(f"  🧪 skill loaded: {self.skill.name}", style="dim"))

        # 短期记忆：AsyncPostgresSaver
        self.checkpointer = asyncio.run(create_checkpointer(DB_URI))
        self.graph = build_graph(checkpointer=self.checkpointer)

        # 长期记忆：AsyncPostgresStore
        self.ltm = LongTermMemory(DB_URI, user_id=self.user_id)

        # LLM（复用 nodes.py 中的单例，避免重复创建连接池）
        self.mem_llm = _get_llm()

        # 后台记忆提取的锁，防止并发写入冲突
        self._mem_lock = threading.Lock()
        # 跟踪上一个后台提取线程，防止与主 graph LLM 并发调用
        self._mem_thread: threading.Thread | None = None

        # 显示会话信息和历史
        history = self._load_history()
        lines = [f"[bold]Tower[/] — Plan→Act→Observe→Refine→Respond",
                 f"Session: [dim]{self.thread_id}[/]"]
        if self.skill:
            lines.append(f"Skill: [cyan]{self.skill.name}[/]")
        if history:
            lines.append("")
            lines.append("[dim]── previous messages ──[/]")
            for role, content in history:
                label = "[bold blue]User:[/]" if role == "user" else "[bold green]Assistant:[/]"
                lines.append(f"  {label} {content[:200]}")
            lines.append("[dim]──[/]")

        console.print(Panel.fit("\n".join(lines), border_style="blue"))

    def _load_history(self) -> list[tuple[str, str]]:
        """从 checkpoint 加载最近的消息历史。"""
        try:
            state = self.graph.get_state(self.config)
            if state is None or state.values is None:
                return []
            msgs = state.values.get("messages", [])
        except Exception:
            return []

        history = []
        for m in msgs[-20:]:  # 最近 20 条
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                continue
            content = m.content if hasattr(m, "content") else ""
            tc = getattr(m, "tool_calls", None)
            if content and (tc is None or len(tc) == 0):
                history.append((role, content))
        return history

    def run(self, user_input: str):
        """执行一轮对话。"""
        # 注入长期记忆（async → asyncio.run 桥接）
        try:
            facts = asyncio.run(self.ltm.get_all_facts())
        except Exception:
            facts = []

        if facts:
            context = "## 关于用户的长期记忆\n"
            context += "\n".join(f"- {f['fact']}" for f in facts)
            task = f"{context}\n\n## 当前任务\n{user_input}"
            console.print(Text(f"  📚 {len(facts)} fact(s) loaded from long-term memory", style="dim"))
        else:
            task = user_input

        # 等待上一轮的后台记忆提取完成，避免与主 graph LLM 并发调用
        if self._mem_thread and self._mem_thread.is_alive():
            self._mem_thread.join(timeout=10)

        # 注入 skill 到 _runtime
        runtime = {}
        if self.skill:
            runtime["skill"] = self.skill

        try:
            # 显式重置 per-task 状态
            input_data = {
                "task": task,
                "pass_count": 0,
                "plan": [],
                "tool_results": {},
                "current_step_index": 0,
                "task_complete": False,
                "node_history": [],
                "_runtime": runtime,
            }
            # 循环处理 LangGraph interrupt（审批、后台任务等）
            while True:
                result = self.graph.invoke(input_data, self.config)
                interrupt_list = result.get("__interrupt__", [])
                if not interrupt_list:
                    break  # 正常完成
                intr_obj = interrupt_list[0]
                intr_value = intr_obj.value if hasattr(intr_obj, "value") else intr_obj

                if isinstance(intr_value, dict) and intr_value.get("type") == "approval":
                    # 审批中断
                    console.print()
                    console.print(Panel(
                        f"[bold]{intr_value['tool']}[/]\n[dim]{intr_value['args']}[/]",
                        title="[bold yellow]⚠ Approval Required[/]",
                        border_style="yellow",
                        padding=(1, 2),
                    ))
                    answer = console.input("  [dim]\\[y/N]:[/] ").strip().lower()
                    approved = answer in ("y", "yes")
                    input_data = Command(resume=approved)

                elif isinstance(intr_value, dict) and intr_value.get("type") == "background_task":
                    # 后台任务中断 — 轮询等待完成
                    task_id = intr_value.get("task_id", "unknown")
                    console.print()
                    console.print(Panel(
                        f"Task ID: [bold cyan]{task_id}[/]\n"
                        f"Status: [yellow]{intr_value.get('status', 'running')}[/]",
                        title="[bold blue]⏳ Background Task[/]",
                        border_style="blue",
                    ))
                    console.print("  [dim]Polling for completion...[/]")

                    # 轮询任务状态（实际实现应查询 MCP/db）
                    import time
                    poll_interval = 2
                    max_polls = 300  # 10 分钟超时
                    for _ in range(max_polls):
                        time.sleep(poll_interval)
                        break  # TODO: 替换为实际任务状态查询

                    # Resume graph with completed result
                    input_data = Command(resume={
                        "status": "completed",
                        "output": "computation completed",
                        "task_id": task_id,
                    })

                else:
                    raise RuntimeError(f"Unknown interrupt: {intr_value}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/]")
            return
        except Exception as e:
            console.print(Panel(str(e), title="Error", border_style="red"))
            traceback.print_exc()
            return

        # 输出回答 —— 用水平线框住，视觉美观且复制时不带边框
        console.print()
        console.print(Rule(style="green"))
        console.print(Markdown(result["final_response"]))
        console.print(Rule(style="green"))

        path = " → ".join(result["node_history"])
        console.print(Text(f"  path: {path}", style="dim"))

        # 长期记忆提取（后台线程，不阻塞用户看到下一个 prompt）
        # 使用锁防止快速连续消息导致的后台线程重叠写入
        import threading
        # 如果上一个提取线程还在运行，等待最多 5 秒后放弃
        if self._mem_thread and self._mem_thread.is_alive():
            self._mem_thread.join(timeout=5)
        def _safe_extract():
            with self._mem_lock:
                try:
                    asyncio.run(extract_memories(
                        user_input, result["final_response"],
                        self.mem_llm, self.ltm,
                    ))
                except Exception:
                    pass  # 后台线程错误不影响主循环
        self._mem_thread = threading.Thread(target=_safe_extract, daemon=True)
        self._mem_thread.start()

        console.print()


@click.group()
def cli():
    """Tower — 基于 LangGraph 的 Plan→Act→Observe→Refine→Respond agent 框架。"""
    pass


@cli.command()
@click.option("--session", "-s", default=None, help="恢复指定会话 ID")
@click.option("--skill-path", "-k", default=None, help="Skill pack 路径（如 skills/dmrg）")
def chat(session, skill_path):
    """启动交互式对话。"""

    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

    chat_session = TowerChat(thread_id=session, user_id=os.getenv("USER", "default"),
                             skill_path=skill_path)

    if session:
        console.print(f"[green]Resumed session: {session}[/]")

    history_dir = ".tower"
    os.makedirs(history_dir, exist_ok=True)
    prompt_session = PromptSession(
        history=FileHistory(os.path.join(history_dir, "input_history.txt")),
    )

    console.print("[dim]/quit 退出  /clear 新会话  /session 查看会话ID[/]\n")

    interrupt_count = 0

    while True:
        try:
            user_input = prompt_session.prompt(
                [("class:prompt", "> ")],
                auto_suggest=AutoSuggestFromHistory(),
            )
            interrupt_count = 0  # 正常输入后重置
        except KeyboardInterrupt:
            interrupt_count += 1
            if interrupt_count == 1:
                console.print("\n  [yellow]Ctrl+C — press again to exit, or type /quit[/]")
                continue
            else:
                console.print()
                break
        except EOFError:
            console.print()
            break

        stripped = user_input.strip()

        if stripped.startswith("/"):
            cmd = stripped[1:].lower()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "clear":
                chat_session = TowerChat(user_id=os.getenv("USER", "default"))
                console.print("[dim]/quit 退出  /clear 新会话[/]\n")
                interrupt_count = 0
                continue
            elif cmd == "session":
                console.print(f"Session: [bold]{chat_session.thread_id}[/]")
                continue
            elif cmd == "help":
                console.print("  /quit   退出")
                console.print("  /clear  开始新会话")
                console.print("  /session  显示会话 ID")
                continue
            else:
                console.print(f"  Unknown command: /{cmd}")
                continue

        if not stripped:
            continue

        chat_session.run(user_input)

    # 退出提示
    console.print()
    console.print(Panel.fit(
        f"Session saved: [bold cyan]{chat_session.thread_id}[/]\n"
        f"Resume with: [dim]tower chat --session {chat_session.thread_id}[/]",
        border_style="yellow",
    ))


@cli.command("sessions")
def list_sessions():
    """列出可恢复的历史会话。"""
    import psycopg
    from rich.table import Table

    try:
        conn = psycopg.connect(DB_URI)
        rows = conn.execute(
            "SELECT thread_id, COUNT(*) as steps "
            "FROM checkpoints WHERE thread_id IS NOT NULL "
            "GROUP BY thread_id ORDER BY MIN(checkpoint_id) DESC LIMIT 10"
        ).fetchall()
        conn.close()
    except Exception as e:
        console.print(f"[red]Failed to query sessions: {e}[/]")
        return

    if not rows:
        console.print("[dim]No past sessions found.[/]")
        return

    table = Table(title="Past Sessions")
    table.add_column("Thread ID", style="cyan")
    table.add_column("Steps", justify="right")

    for thread_id, steps in rows:
        table.add_row(thread_id, str(steps))

    console.print(table)
    console.print("\n[dim]Resume with: tower chat --session <thread-id>[/]")


@cli.command()
@click.argument("task")
@click.option("--session", "-s", default=None, help="会话 ID")
@click.option("--skill-path", "-k", default=None, help="Skill pack 路径（如 skills/dmrg）")
def run(task: str, session: str | None, skill_path: str | None):
    """执行单个任务。"""

    chat_session = TowerChat(thread_id=session, user_id=os.getenv("USER", "default"),
                             skill_path=skill_path)
    chat_session.run(task)


def main():
    cli()


if __name__ == "__main__":
    main()
