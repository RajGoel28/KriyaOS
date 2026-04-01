# ============================================================
#  KriyaOS — orchestrator/agents/design_agent.py
#  Generates UI layouts, component structures, and design specs.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.ai_core import ask, stream
from core.memory import save_task
from orchestrator.bus import bus, Topic
from orchestrator.model_manager import ensure_loaded
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are Kriya's Design Agent inside KriyaOS.
You design UI components and layouts for React + Tailwind CSS.
Always output:
1. Component structure (which components are needed)
2. Layout description (how they're arranged)
3. React + Tailwind code for the main component
4. Color and spacing notes

Use KriyaOS color palette:
- Background: #0F0F14
- Surface: #1A1A24
- Purple accent: #7F77DD
- Teal success: #1D9E75
- Text primary: #E8E6FF"""


def design_component(description: str, streaming: bool = True) -> str:
    """
    Design a React component from a description.

    Usage:
        component = design_component("A dashboard sidebar with model status indicators")
    """
    ensure_loaded(["planner"])
    console.print("\n[bold cyan][ Design Agent ][/bold cyan]")

    prompt = f"Design and implement this UI component:\n\n{description}"

    if streaming:
        result = ""
        for chunk in stream(prompt, role="planner", system_prompt=SYSTEM_PROMPT):
            result += chunk
    else:
        result = ask(prompt, role="planner", system_prompt=SYSTEM_PROMPT)
        console.print(result)

    bus.post(Topic.RESULT_DESIGN, "design_agent", {"prompt": description, "design": result})
    save_task("design", prompt, result, "planner")
    return result