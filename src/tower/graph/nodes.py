"""Tower graph nodes — Plan→Act→Observe→Refine→Respond。"""
import json
import re
import time
import threading
from contextlib import contextmanager
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from rich.console import Console
from rich.panel import Panel
from tower.state import AgentState
from tower.tools.registry import TOOLS, TOOL_BY_NAME

console = Console()

# ═══════════════════════════════════════════════════════════════════
# Animated thinking spinner
# ═══════════════════════════════════════════════════════════════════

# npm-like braille spinner frames
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_INTERVAL = 0.08  # 秒


@contextmanager
def _thinking_spinner(label: str = "thinking"):
    """npm 风格的绿色动态旋转指示器，在 LLM 调用期间显示。"""
    stop_spin = threading.Event()
    start_time = time.time()

    def spin():
        i = 0
        while not stop_spin.is_set():
            elapsed = time.time() - start_time
            frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
            console.print(
                f"  [bold green]{frame}[/] [green]{label}[/] [dim]({elapsed:.0f}s)[/]",
                end="\r",
            )
            i += 1
            time.sleep(SPINNER_INTERVAL)
        # 清除整行
        console.print(" " * 60, end="\r")

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_spin.set()
        t.join(timeout=0.5)

_llm = None
_model_name = "deepseek:deepseek-v4-flash"


def set_model(name: str):
    """切换 LLM 模型（必须在使用任何 node 之前调用）。"""
    global _llm, _model_name
    _model_name = name
    _llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = init_chat_model(_model_name, timeout=45, max_retries=1)
    return _llm


def _get_llm_with_tools():
    return _get_llm().bind_tools(TOOLS)


def _sanitize_messages(msgs: list) -> list:
    """清理消息列表，移除孤立的 tool_calls 和 ToolMessage（DeepSeek API 要求）。

    DeepSeek 的两条硬约束：
    1. AIMessage 中的每个 tool_call_id 后面必须有对应的 ToolMessage。
    2. 每个 ToolMessage 必须是某个前置 AIMessage 中 tool_call 的响应。

    如果 plan_node 一次性产生了多条 tool_call（如 4 条），但 act_node
    只执行了部分就进入 refine，则未执行的那些 tool_call 缺少 ToolMessage
    （违反规则 1）。如果历史窗口截断了前面的 AIMessage 但保留了后面的
    ToolMessage，则 ToolMessage 成为孤儿（违反规则 2）。

    策略：双向清理 ——
    - AIMessage：只保留有对应 ToolMessage 的 tool_call
    - ToolMessage：只保留 tool_call_id 出现在某个 AIMessage 中的
    """
    # 收集所有 ToolMessage 的 tool_call_id
    responded_ids: set[str] = set()
    for m in msgs:
        if isinstance(m, ToolMessage):
            tc_id = getattr(m, "tool_call_id", None)
            if tc_id:
                responded_ids.add(tc_id)

    # 收集所有 AIMessage 声明过的 tool_call_id
    declared_ids: set[str] = set()
    for m in msgs:
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None)
            if tc:
                for t in tc:
                    t_id = t.get("id", "")
                    if t_id:
                        declared_ids.add(t_id)

    cleaned = []
    for m in msgs:
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None)
            if tc is None:
                cleaned.append(m)
                continue
            if len(tc) == 0:
                # 空 tool_calls → 去掉 tool_calls 字段但保留所有其他属性
                cleaned.append(m.model_copy(update={"tool_calls": []}))
                continue
            # 只保留有对应 ToolMessage 的 tool_calls（满足规则 1）
            valid_tc = [
                t for t in tc
                if t.get("id", "") in responded_ids
            ]
            if valid_tc:
                cleaned.append(m.model_copy(update={"tool_calls": valid_tc}))
            else:
                # 全部都没有 ToolMessage → 去掉 tool_calls 字段
                cleaned.append(m.model_copy(update={"tool_calls": []}))
        elif isinstance(m, ToolMessage):
            # 只保留 tool_call_id 出现在某个 AIMessage 中的 ToolMessage（满足规则 2）
            tc_id = getattr(m, "tool_call_id", "")
            if tc_id and tc_id in declared_ids:
                cleaned.append(m)
            # 否则丢弃这个孤儿 ToolMessage
        else:
            cleaned.append(m)
    return cleaned


# ============================================================
# System Prompts
# ============================================================

SYSTEM_PLAN = """你是一个能使用工具的 AI agent。

## 工作目录
- 当前工作目录是项目根目录，所有操作默认在此目录下执行
- 除非用户明确指定其他路径，否则 path 参数只传文件名或相对路径
- 不要跳到 /tmp、~/.config 等外部目录，除非用户明确要求

## 安全规则
- 你只能在当前项目目录下操作，不能访问系统目录（/etc、/usr 等）
- 不能以 root 权限执行命令
- 删除文件需要用户确认

## 可用工具
| 工具 | 用途 |
|------|------|
| read_file | 读取文件内容 |
| write_file | 创建/覆写文件 |
| edit_file | 精确替换文件中的字符串（old_string → new_string） |
| list_directory | 列出目录内容（文件/夹/大小） |
| glob_tool | 按模式查找文件（如 "**/*.py"） |
| grep | 在文件中搜索文本/正则 |
| bash | 执行 shell 命令（编译、运行、git 等） |
| move_file | 移动/重命名文件 |
| copy_file | 复制文件 |
| delete_file | 删除文件（需确认） |
| web_fetch | 抓取网页内容 |
| web_search | 网络搜索 |

## 规则
1. 闲聊/问候/简单问答 → 直接回复，不调用工具
2. 编码任务 → 一次性规划所有步骤（如：读代码 → 编辑 → 编译验证）
3. 简单明确的任务直接执行，不要反复搜索确认。比如"删除 X 重写 Y"：直接写新代码即可，不需要先搜三遍文件在哪
4. 使用 edit_file 做精确修改，不要用 write_file 重写整个文件
5. 信息收集最多 1-2 步就得开始执行，不要陷入搜索循环
6. 执行代码用 bash + python -c 或 bash + 编译器命令

## 示例
- "hello" → 直接回复
- "列出文件" → list_directory(".")
- "计算斐波那契" → bash(command="python -c '...'")
- "修复 main.py 的 bug" → read_file + edit_file + bash 验证
- "删除 X 重写 Y" → write_file 写新代码 + bash 编译，不要先搜 fib 文件"""

SYSTEM_REFINE = """工具执行失败了。分析错误，决定下一步：
- 如果能用其他方式解决，调用修正后的工具
- 如果无法解决，直接回复，说明原因"""

SYSTEM_CONTINUE = """你是能使用工具的 AI agent。

根据已执行的操作结果，判断用户的原始任务是否**真正**完成。

工作目录：所有路径操作默认在当前项目目录（"."），不要跳到外部目录。

关键判断标准：
- 如果用户要求"删除"/"创建"/"修改"/"安装"/"运行"等操作，仅仅 ls/cat/read_file 等信息收集
  不算完成任务——你必须继续调用工具来执行实际操作。
- 信息收集（ls、read_file 等）只是准备工作，不是任务的终点。
- 只有当用户的原始请求被完全满足时，才可以直接回复。

如果还需要操作：调用工具
如果任务已真正完成：直接回复（不要调用工具）

可用工具：read_file、write_file、edit_file、list_directory、glob_tool、grep、
          bash、move_file、copy_file、delete_file、web_fetch、web_search"""

MAX_PASSES = 5

SYSTEM_RESPOND = """你是用户的 AI 助手。

请用纯自然语言回答用户。像一个有礼貌的助手那样：
- 先说结论或结果
- 如果需要，补充关键细节
- 如果有失败，诚实说明

重要：你现在没有工具可用，只需要总结已经发生的事情。不要输出 tool_calls 或
类似 <｜｜DSML｜｜tool_calls> 的 XML 标签，那是给系统看的，不是给用户看的。"""


# ============================================================
# Approval
# ============================================================

# 需要用户确认才能执行的操作
APPROVAL_REQUIRED = {"bash", "write_file", "delete_file", "move_file"}


def _check_approval(tool_name: str, args: dict) -> bool:
    """检查是否需要用户确认。返回 True 表示已批准（或不需要批准）。

    使用 LangGraph interrupt() 暂停 graph 执行，等待用户审批。
    这样 checkpoint 会在暂停前保存，避免进程崩溃时丢失状态。
    """
    if tool_name not in APPROVAL_REQUIRED:
        return True

    # 格式化参数，传递给主循环用于显示
    args_str = json.dumps(args, ensure_ascii=False, indent=2)
    if len(args_str) > 600:
        args_str = args_str[:600] + "\n  ..."

    # 暂停 graph，主循环处理 UI 交互后通过 Command(resume=...) 恢复
    approved = interrupt({
        "type": "approval",
        "tool": tool_name,
        "args": args_str,
    })
    return bool(approved)


def _strip_tool_call_xml(text: str) -> str:
    """移除 LLM 在纯文本回复中幻觉出的 tool_calls / invoke XML 片段。"""
    # 去掉零宽空格
    text = text.replace('​', '')
    # 移除 <function_calls>...</function_calls> 整块（包含内容）
    text = re.sub(r'<\s*function_calls\s*>.*?</\s*function_calls\s*>', '', text, flags=re.DOTALL)
    # 移除单独的 <invoke>, <parameter> 标签
    text = re.sub(r'</?\s*(invoke|parameter)\s*[^>]*/?>', '', text)
    # 清理残留空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================
# Nodes
# ============================================================


def plan_node(state: AgentState) -> dict:
    """分析任务：首次生成计划；或上一轮工具执行完后延续规划。

    pass_count == 0 → 首次规划（重置 per-task 字段）
    pass_count > 0  → 延续模式（保留已有结果，追加新步骤或标记完成）
    """
    llm = _get_llm_with_tools()
    history = list(state.get("messages", []))
    pass_count = state.get("pass_count", 0)

    # ── 延续模式：基于已有结果判断是否还需要更多工具 ──
    if pass_count > 0:
        if pass_count > MAX_PASSES:
            print(f"\n\033[2m[PLAN] max passes ({MAX_PASSES}) reached — forcing respond\033[0m")
            return {
                "task_complete": True,
                "node_history": state.get("node_history", []) + ["plan"],
            }

        results = state.get("tool_results", {})
        existing_plan = list(state.get("plan", []))

        results_text = "\n".join(
            f"[{k}] {json.dumps(v, ensure_ascii=False)[:500]}"
            for k, v in sorted(results.items())
        )

        continue_msg = HumanMessage(content=(
            f"已执行的操作结果：\n{results_text}\n\n"
            f"用户的原始任务：{state['task']}\n\n"
            f"请严格判断：用户要求做的事是否已经**全部**完成了？\n"
            f"- 如果用户要求删除/创建/修改/运行等操作，而目前只做了 ls/cat 等信息收集，"
            f"那你必须继续调用工具来完成实际操作。\n"
            f"- 只有当原始任务被完全满足时，才直接回复。\n"
            f"- 所有操作默认在当前目录（.）下执行，不要跳到 /tmp 等外部目录。"
        ))

        try:
            with _thinking_spinner("planning"):
                response = llm.invoke(
                    [SystemMessage(content=SYSTEM_CONTINUE),
                     *_sanitize_messages(history[-20:]),
                     continue_msg]
                )
        except Exception as e:
            print(f"\n\033[33m[PLAN] LLM call failed in continue mode: {e}\033[0m")
            return {
                "task_complete": True,
                "final_response": f"LLM 调用失败，无法继续处理任务：{e}",
                "messages": [continue_msg],
                "node_history": state.get("node_history", []) + ["plan"],
            }

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            print("\n\033[2m[PLAN] task complete — no more tools needed\033[0m")
            # 如果 LLM 在判断完成时附带了总结，直接作为 final_response 传递给 respond_node
            # 避免 respond_node 再做一次 LLM 调用来生成相同的总结
            direct_reply = response.content.strip() if hasattr(response, "content") and response.content else ""
            result = {
                "task_complete": True,
                "messages": [continue_msg, response],
                "node_history": state.get("node_history", []) + ["plan"],
            }
            if direct_reply:
                result["final_response"] = direct_reply
            return result

        new_plan = existing_plan.copy()
        print(f"\n[PLAN] +{len(tool_calls)} more tool call(s):")
        for tc in tool_calls:
            new_plan.append({
                "name": tc["name"],
                "args": tc["args"],
                "id": tc.get("id", ""),
            })
            print(f"  {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")

        return {
            "plan": new_plan,
            "current_step_index": len(existing_plan),
            "pass_count": pass_count + 1,
            "refinement_needed": False,
            "messages": [continue_msg, response],
            "node_history": state.get("node_history", []) + ["plan"],
        }

    # ── 首次规划 ──
    task_msg = HumanMessage(content=state["task"])
    context = history + [task_msg]
    try:
        with _thinking_spinner("planning"):
            response = llm.invoke(
                [SystemMessage(content=SYSTEM_PLAN), *_sanitize_messages(context[-30:])]
            )
    except Exception as e:
        print(f"\n\033[33m[PLAN] LLM call failed: {e}\033[0m")
        return {
            "task_complete": True,
            "plan": [],
            "final_response": f"LLM 调用失败：{e}",
            "messages": [task_msg],
            "node_history": ["plan"],
        }

    new_msgs = [task_msg, response]
    tool_calls = getattr(response, "tool_calls", None) or []

    base = {
        "current_step_index": 0,
        "tool_results": {},
        "observation": "",
        "refinement_needed": False,
        "retry_pending": False,
        "retry_count": 0,
        "pass_count": 1,
        "final_response": "",
        "task_complete": False,
        "node_history": ["plan"],
    }

    if not tool_calls:
        print("\n\033[2m[PLAN] conversational — will respond directly\033[0m")
        return {**base, "plan": [], "messages": new_msgs}

    plan = []
    print(f"\n[PLAN] {len(tool_calls)} tool call(s):")
    for tc in tool_calls:
        plan.append({
            "name": tc["name"],
            "args": tc["args"],
            "id": tc.get("id", ""),
        })
        print(f"  {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")

    return {**base, "plan": plan, "messages": new_msgs}


def _execute_tool(tc: dict, step_key: str, step_num: int, total: int) -> tuple[dict, ToolMessage]:
    """执行单个 tool call，返回 (result, tool_message)。"""
    tool_name = tc["name"]

    print(f"\n[ACT] {step_num}/{total}: {tool_name}({json.dumps(tc['args'], ensure_ascii=False)})")

    # 审批检查
    if not _check_approval(tool_name, tc["args"]):
        result = {"error": "User denied approval"}
        tool_msg = ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=tc.get("id", ""),
        )
        print(f"  DENIED by user")
        return result, tool_msg

    tool_fn = TOOL_BY_NAME.get(tool_name)
    if tool_fn is None:
        result = {"error": f"Unknown tool: {tool_name}"}
    else:
        try:
            result = tool_fn.invoke(tc["args"])
        except Exception as e:
            result = {"error": str(e)}

    serialized = json.dumps(result, ensure_ascii=False)
    max_chars = 10000
    if len(serialized) > max_chars:
        half = max_chars // 2
        serialized = (
            serialized[:half]
            + f"\n... [truncated, {len(serialized) - max_chars} more chars in middle] ...\n"
            + serialized[-half:]
        )
        print(f"  [WARN] tool output truncated (head+tail preserved, {len(serialized.split(chr(10))[0])} chars total)")

    tool_msg = ToolMessage(
        content=serialized,
        tool_call_id=tc.get("id", ""),
    )

    if "error" in result:
        print(f"  FAILED: {result['error'][:200]}")
    elif "stdout" in result:
        print(f"  stdout: {result['stdout'][:300]}")
    elif "content" in result:
        print(f"  content: {result['content'][:300]}")
    elif "ok" in result:
        print(f"  {result['ok']}")
    else:
        print(f"  result: {json.dumps(result, ensure_ascii=False)[:200]}")

    return result, tool_msg


def act_node(state: AgentState) -> dict:
    """执行 plan 中 current_step_index 指向的 tool call。

    如果 retry_pending=True（refine 触发的重试），执行上一步但不回退 index，
    避免 current_step_index 双向移动造成的混乱状态追踪。
    """
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    results = dict(state.get("tool_results", {}))
    retry_pending = state.get("retry_pending", False)

    # refine 触发的重试：执行上一步，不回退 current_step_index
    if retry_pending:
        retry_step = idx - 1
        if retry_step < 0 or retry_step >= len(plan):
            return {"retry_pending": False, "node_history": state.get("node_history", []) + ["act"]}

        tc = plan[retry_step]
        step_key = f"step_{retry_step}"
        result, tool_msg = _execute_tool(tc, step_key, retry_step + 1, len(plan))
        results[step_key] = result

        return {
            "tool_results": results,
            "messages": [tool_msg],
            "current_step_index": idx,  # 不回退，保持正向移动
            "retry_pending": False,
            "node_history": state.get("node_history", []) + ["act"],
        }

    if idx >= len(plan):
        return {"node_history": state.get("node_history", []) + ["act"]}

    tc = plan[idx]
    step_key = f"step_{idx}"
    result, tool_msg = _execute_tool(tc, step_key, idx + 1, len(plan))
    results[step_key] = result

    return {
        "tool_results": results,
        "messages": [tool_msg],
        "current_step_index": idx + 1,
        "node_history": state.get("node_history", []) + ["act"],
    }


def observe_node(state: AgentState) -> dict:
    """检查上一步工具执行结果。"""
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    results = state.get("tool_results", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    last_key = f"step_{idx - 1}" if idx > 0 else None
    last_result = results.get(last_key, {}) if last_key else {}
    tool_name = plan[idx - 1]["name"] if idx > 0 and idx <= len(plan) else "?"

    print(f"\n[OBSERVE] step {idx}/{len(plan)} — {tool_name}")

    # 判断失败：只看 "error" 字段，不做 returncode 的默认值假设
    if last_result.get("error"):
        err = last_result["error"]
        observation = f"Tool '{tool_name}' failed: {err}"
        refinement_needed = retry_count < max_retries
        print(f"  FAILED ({'will retry' if refinement_needed else 'giving up'})")
    elif tool_name == "bash" and "returncode" in last_result and last_result["returncode"] != 0:
        # bash 特有的 returncode 检查 —— 仅当 returncode 字段确实存在且非 0 时才判失败
        rc = last_result["returncode"]
        stderr = last_result.get("stderr", "")
        observation = f"Tool 'bash' failed (rc={rc}): {stderr}"
        refinement_needed = retry_count < max_retries
        print(f"  FAILED ({'will retry' if refinement_needed else 'giving up'})")
    elif tool_name == "bash" and "returncode" in last_result and last_result["returncode"] == 0:
        # bash 明确 returncode=0 —— 成功
        stderr = last_result.get("stderr", "")
        if stderr:
            print(f"  OK (rc=0, stderr has {len(stderr)} chars)")
        else:
            print("  OK")
        observation = f"Tool 'bash' succeeded"
        refinement_needed = False
    else:
        observation = f"Tool '{tool_name}' succeeded"
        refinement_needed = False
        print("  OK")

    return {
        "observation": observation,
        "refinement_needed": refinement_needed,
        "node_history": state.get("node_history", []) + ["observe"],
    }


def refine_node(state: AgentState) -> dict:
    """工具失败后，让 LLM 决定如何修正。"""
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    observation = state.get("observation", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    history = list(state.get("messages", []))

    if retry_count >= max_retries:
        print(f"\n[REFINE] max retries ({max_retries}) reached")
        return {
            "refinement_needed": False,
            "retry_count": max_retries,
            "observation": f"Max retries ({max_retries}) exceeded. Giving up on fixing this step.",
            "node_history": state.get("node_history", []) + ["refine"],
        }

    print(f"\n[REFINE] retry {retry_count + 1}/{max_retries}")

    if idx < 1 or idx > len(plan):
        print(f"  bad idx={idx}, plan len={len(plan)}, skipping refine")
        return {
            "refinement_needed": False,
            "retry_count": max_retries,
            "observation": f"Bad state: step index {idx} out of range (plan has {len(plan)} steps).",
            "node_history": state.get("node_history", []) + ["refine"],
        }

    failed_tc = plan[idx - 1]

    refine_prompt = HumanMessage(content=(
        f"工具 {failed_tc['name']} 执行失败:\n{observation}\n\n"
        f"请分析错误并尝试修正。"
    ))

    llm = _get_llm_with_tools()
    try:
        with _thinking_spinner("refining"):
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_REFINE),
                    *_sanitize_messages(history[-15:]),
                    refine_prompt,
                ]
            )
    except Exception as e:
        print(f"\n  \033[33m[REFINE] LLM call failed: {e}\033[0m")
        return {
            "refinement_needed": False,
            "messages": [refine_prompt],
            "retry_count": max_retries,
            "node_history": state.get("node_history", []) + ["refine"],
        }

    new_tool_calls = getattr(response, "tool_calls", None) or []

    if new_tool_calls:
        # 替换失败步骤为 LLM 建议的新 tool call（只取第一个）
        new_plan = list(plan)
        new_tc = {
            "name": new_tool_calls[0]["name"],
            "args": new_tool_calls[0]["args"],
            "id": new_tool_calls[0].get("id", ""),
        }
        new_plan[idx - 1] = new_tc
        print(f"  corrected: {new_tc['name']}({json.dumps(new_tc['args'], ensure_ascii=False)})")

        return {
            "plan": new_plan,
            "messages": [refine_prompt, response],
            "current_step_index": idx,  # 不回退，通过 retry_pending 触发重试
            "retry_count": retry_count + 1,
            "refinement_needed": False,
            "retry_pending": True,
            "node_history": state.get("node_history", []) + ["refine"],
        }
    else:
        # LLM 放弃了 → 消耗剩余重试次数，避免死循环
        msg = response.content if hasattr(response, "content") else str(response)
        print(f"  gave up: {msg[:200]}")
        return {
            "refinement_needed": False,
            "messages": [refine_prompt, response],
            "retry_count": max_retries,
            "node_history": state.get("node_history", []) + ["refine"],
        }


def respond_node(state: AgentState) -> dict:
    """生成最终回答：有工具结果就汇总，闲聊直接用 plan_node 的回复。

    如果 plan_node（延续模式）已经生成了 final_response，直接复用，
    避免重复 LLM 调用。
    """
    # plan_node 延续模式可能已生成 final_response，直接复用
    # 只有非空的最终回复才跳过总结——空字符串是初始化占位值
    existing_final = (state.get("final_response") or "").strip()
    if existing_final:
        print("\n\033[2m[RESPOND] reusing plan_node response\033[0m")
        return {
            "final_response": _strip_tool_call_xml(existing_final),
            "task_complete": True,
            "node_history": state.get("node_history", []) + ["respond"],
        }

    results = state.get("tool_results", {})
    plan = state.get("plan", [])
    history = list(state.get("messages", []))

    if not results and not plan:
        # 纯闲聊：取 plan_node 最后一条 AIMessage
        final = ""
        for m in reversed(history):
            if isinstance(m, AIMessage):
                final = m.content
                break
            # checkpoint 恢复后消息可能失去类型信息 —— 回退检查 content 属性
            if hasattr(m, "content") and hasattr(m, "type") and m.type == "ai":
                final = m.content
                break

        # 如果遍历完仍为空，返回明确的错误信息而非空白
        if not final:
            final = "（无法生成回复，请重试或检查 LLM 连接）"

        final = _strip_tool_call_xml(final)

        print("\n\033[2m[RESPOND] using plan_node response\033[0m")

        return {
            "final_response": final,
            "task_complete": True,
            "node_history": state.get("node_history", []) + ["respond"],
        }

    print("\n[RESPOND] generating response from tool results...")

    # 汇总所有结果（包括错误）
    lines = []
    for k, v in sorted(results.items()):
        key_info = v.get("command", v.get("path", k))
        if "error" in v:
            lines.append(f"[{k}] {key_info}: ERROR - {v['error']}")
        elif "stdout" in v:
            lines.append(f"[{k}] {key_info}: {v['stdout'][:300]}")
        elif "content" in v:
            lines.append(f"[{k}] {key_info}: {v['content'][:300]}")
        elif "ok" in v:
            lines.append(f"[{k}] {key_info}: {v['ok']}")
        else:
            lines.append(f"[{k}] {key_info}: {json.dumps(v, ensure_ascii=False)[:200]}")

    results_text = "\n".join(lines)

    summarize_prompt = HumanMessage(content=(
        f"请根据以下执行结果回答用户：\n\n{results_text}\n\n"
        f"用户原始请求：{state['task']}"
    ))

    try:
        with _thinking_spinner("responding"):
            response = _get_llm().invoke(
                [
                    SystemMessage(content=SYSTEM_RESPOND),
                    *_sanitize_messages(history[-30:]),
                    summarize_prompt,
                ]
            )
        final = response.content.strip() if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"\n  \033[33m[RESPOND] LLM call failed: {e}\033[0m")
        # 回退：直接返回原始结果摘要
        final = (
            f"任务执行完成，但生成总结时遇到错误：{e}\n\n"
            f"原始结果：\n\n{results_text}"
        )
    final = _strip_tool_call_xml(final)

    print("[RESPOND] done")

    return {
        "final_response": final,
        "task_complete": True,
        "node_history": state.get("node_history", []) + ["respond"],
    }
