# ============================================================
#  KriyaOS — orchestrator/agents/reviewer.py
#  Reviews code for quality, bugs, security, and improvements.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.ai_core import ask, stream
from core.memory import save_task
from orchestrator.bus import bus, Topic
from orchestrator.model_manager import ensure_loaded
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are Kriya's Reviewer Agent inside KriyaOS.
You review code with the eyes of a senior engineer.
Structure every review as:

VERDICT: [PASS / NEEDS WORK / FAIL]

BUGS:
- [any bugs found, or "None"]

SECURITY:
- [any security issues, or "None"]

QUALITY:
- [code quality observations]

IMPROVEMENTS:
- [specific actionable improvements]

SCORE: [X/10]

Be direct and specific. No vague feedback."""


def review(code: str, context: str = "", streaming: bool = True) -> str:
    """
    Review code and return structured feedback.

    Args:
        code:    The code to review
        context: Optional context about what the code is supposed to do

    Usage:
        feedback = review("def login(user, pwd): return user == pwd")
    """
    ensure_loaded(["orchestrator"])
    console.print("\n[bold cyan][ Reviewer Agent ][/bold cyan]")

    review_prompt = f"Review this code:\n\n```python\n{code}\n```"
    if context:
        review_prompt += f"\n\nContext: {context}"

    if streaming:
        result = ""
        for chunk in stream(review_prompt, role="orchestrator", system_prompt=SYSTEM_PROMPT):
            result += chunk
    else:
        result = ask(review_prompt, role="orchestrator", system_prompt=SYSTEM_PROMPT)
        console.print(result)

    bus.post(Topic.RESULT_REVIEW, "reviewer", {"code": code, "review": result})
    save_task("review", review_prompt, result, "orchestrator")
    return result