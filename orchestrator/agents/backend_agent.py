# ============================================================
#  KriyaOS — orchestrator/agents/backend_agent.py
#  Handles all code tasks — write, fix, explain.
#  Uses Qwen2.5-Coder as the main model.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.ai_core import ask, stream
from core.memory import save_task
from orchestrator.bus import bus, Topic
from orchestrator.model_manager import ensure_loaded
from rich.console import Console

console = Console()

WRITE_SYSTEM = """You are Kriya's Backend Agent inside KriyaOS.
You are an expert Python developer. Write clean, production-ready code.
Always include:
- Type hints
- Docstrings
- Error handling
- Example usage in comments
Never write placeholder code. Always write the real implementation."""

FIX_SYSTEM = """You are Kriya's Backend Agent inside KriyaOS.
You are an expert Python debugger. When given broken code:
1. Identify the exact bug
2. Explain why it's happening
3. Show the fixed code
4. Explain what you changed
Be precise and direct."""

EXPLAIN_SYSTEM = """You are Kriya's Backend Agent inside KriyaOS.
You explain code clearly to developers of all levels.
Break down what the code does step by step.
Point out any patterns, potential issues, or improvements."""


def write(prompt: str, streaming: bool = True) -> str:
    """
    Write code for a given task.

    Usage:
        code = write("Write a FastAPI endpoint for user login")
    """
    ensure_loaded(["coder"])
    console.print("\n[bold cyan][ Backend Agent — Write ][/bold cyan]")

    if streaming:
        result = ""
        for chunk in stream(prompt, role="coder", system_prompt=WRITE_SYSTEM):
            result += chunk
    else:
        result = ask(prompt, role="coder", system_prompt=WRITE_SYSTEM)
        console.print(result)

    bus.post(Topic.RESULT_CODE, "backend_agent", {"prompt": prompt, "code": result})
    save_task("code", prompt, result, "coder")
    return result


def fix(code: str, error: str = "", streaming: bool = True) -> str:
    """
    Fix broken code.

    Args:
        code:  The broken code
        error: The error message (optional but helpful)

    Usage:
        fixed = fix("def sort(lst): return lst.sort()", "returns None instead of list")
    """
    ensure_loaded(["coder"])
    console.print("\n[bold cyan][ Backend Agent — Fix ][/bold cyan]")

    fix_prompt = f"Fix this code:\n\n```python\n{code}\n```"
    if error:
        fix_prompt += f"\n\nError message:\n{error}"

    if streaming:
        result = ""
        for chunk in stream(fix_prompt, role="coder", system_prompt=FIX_SYSTEM):
            result += chunk
    else:
        result = ask(fix_prompt, role="coder", system_prompt=FIX_SYSTEM)
        console.print(result)

    bus.post(Topic.RESULT_FIX, "backend_agent", {"code": code, "fix": result})
    save_task("fix", fix_prompt, result, "coder")
    return result


def explain(code: str, streaming: bool = True) -> str:
    """
    Explain what a piece of code does.

    Usage:
        explanation = explain("lambda x: x if x > 0 else -x")
    """
    ensure_loaded(["coder"])
    console.print("\n[bold cyan][ Backend Agent — Explain ][/bold cyan]")

    explain_prompt = f"Explain this code:\n\n```python\n{code}\n```"

    if streaming:
        result = ""
        for chunk in stream(explain_prompt, role="coder", system_prompt=EXPLAIN_SYSTEM):
            result += chunk
    else:
        result = ask(explain_prompt, role="coder", system_prompt=EXPLAIN_SYSTEM)
        console.print(result)

    save_task("explain", explain_prompt, result, "coder")
    return result