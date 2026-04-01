# ============================================================
#  KriyaOS — orchestrator/agents/testing_agent.py
#  Writes pytest test cases for any Python code.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.ai_core import ask, stream
from core.memory import save_task
from orchestrator.bus import bus, Topic
from orchestrator.model_manager import ensure_loaded
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are Kriya's Testing Agent inside KriyaOS.
You write comprehensive pytest test suites.
For every function, always write tests for:
- Happy path (normal input)
- Edge cases (empty, zero, None, boundary values)
- Error cases (invalid input, exceptions)

Use pytest fixtures where appropriate.
Always include a docstring explaining what each test checks."""


def write_tests(code: str, context: str = "", streaming: bool = True) -> str:
    """
    Write pytest tests for a piece of code.

    Args:
        code:    The code to write tests for
        context: Optional description of what the code does

    Usage:
        tests = write_tests("def divide(a, b): return a / b")
    """
    ensure_loaded(["coder"])
    console.print("\n[bold cyan][ Testing Agent ][/bold cyan]")

    prompt = f"Write comprehensive pytest tests for this code:\n\n```python\n{code}\n```"
    if context:
        prompt += f"\n\nContext: {context}"

    if streaming:
        result = ""
        for chunk in stream(prompt, role="coder", system_prompt=SYSTEM_PROMPT):
            result += chunk
    else:
        result = ask(prompt, role="coder", system_prompt=SYSTEM_PROMPT)
        console.print(result)

    save_task("test", prompt, result, "coder")
    return result