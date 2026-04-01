# ============================================================
#  KriyaOS — orchestrator/agents/planner.py
#  Breaks complex tasks into step-by-step execution plans.
#  Called first in full_team pipeline before other agents run.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.ai_core import ask, stream
from core.memory import save_task
from orchestrator.bus import bus, Topic
from orchestrator.model_manager import ensure_loaded
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are Kriya's Planner agent inside KriyaOS.
Your job is to break any complex task into a clear, numbered execution plan.
Each step must be concrete and actionable.
Format your output as:

PLAN:
1. [step one]
2. [step two]
...

AGENTS NEEDED:
- [list which agents should handle which steps]

Keep it concise. No fluff."""


def run(prompt: str, streaming: bool = True) -> str:
    """
    Break a complex task into an execution plan.

    Args:
        prompt:    The complex task to plan
        streaming: Stream output live

    Returns:
        The plan as a string

    Usage:
        plan = run("Build a full e-commerce API with auth and payments")
    """
    ensure_loaded(["planner"])

    planning_prompt = f"Create a detailed execution plan for this task:\n\n{prompt}"

    console.print("\n[bold cyan][ Planner Agent ][/bold cyan]")

    if streaming:
        result = ""
        for chunk in stream(planning_prompt, role="planner", system_prompt=SYSTEM_PROMPT):
            result += chunk
    else:
        result = ask(planning_prompt, role="planner", system_prompt=SYSTEM_PROMPT)
        console.print(result)

    bus.post(Topic.RESULT_PLAN, "planner", {"prompt": prompt, "plan": result})
    save_task("plan", prompt, result, "planner")

    return result