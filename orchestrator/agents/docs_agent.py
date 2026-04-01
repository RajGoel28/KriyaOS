# ============================================================
#  KriyaOS — orchestrator/agents/docs_agent.py
#  Writes documentation, READMEs, docstrings, and summaries.
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.ai_core import ask, stream
from core.memory import save_task
from orchestrator.bus import bus, Topic
from orchestrator.model_manager import ensure_loaded
from rich.console import Console

console = Console()

DOCSTRING_SYSTEM = """You are Kriya's Docs Agent inside KriyaOS.
Write clear, professional Python docstrings in Google style.
Include: summary, Args, Returns, Raises, Example."""

README_SYSTEM = """You are Kriya's Docs Agent inside KriyaOS.
Write clean, professional README.md files.
Include: project title, description, features, installation,
usage examples, and license section."""

SUMMARY_SYSTEM = """You are Kriya's Docs Agent inside KriyaOS.
Summarize code or technical content clearly and concisely.
Target audience: developers. Be precise, not verbose."""


def write_docstring(code: str, streaming: bool = True) -> str:
    """
    Write docstrings for a Python function or class.

    Usage:
        docs = write_docstring("def add(a, b): return a + b")
    """
    ensure_loaded(["docs"])
    console.print("\n[bold cyan][ Docs Agent — Docstring ][/bold cyan]")

    prompt = f"Write a complete docstring for this Python code:\n\n```python\n{code}\n```"

    if streaming:
        result = ""
        for chunk in stream(prompt, role="docs", system_prompt=DOCSTRING_SYSTEM):
            result += chunk
    else:
        result = ask(prompt, role="docs", system_prompt=DOCSTRING_SYSTEM)
        console.print(result)

    save_task("docs", prompt, result, "docs")
    return result


def write_readme(project_description: str, streaming: bool = True) -> str:
    """
    Write a README.md for a project.

    Usage:
        readme = write_readme("KriyaOS is an AI-native operating platform...")
    """
    ensure_loaded(["docs"])
    console.print("\n[bold cyan][ Docs Agent — README ][/bold cyan]")

    prompt = f"Write a complete README.md for this project:\n\n{project_description}"

    if streaming:
        result = ""
        for chunk in stream(prompt, role="docs", system_prompt=README_SYSTEM):
            result += chunk
    else:
        result = ask(prompt, role="docs", system_prompt=README_SYSTEM)
        console.print(result)

    bus.post(Topic.RESULT_DOCS, "docs_agent", {"prompt": prompt, "docs": result})
    save_task("docs", prompt, result, "docs")
    return result


def summarize(content: str, streaming: bool = True) -> str:
    """
    Summarize a piece of code or technical document.

    Usage:
        summary = summarize(long_code_file)
    """
    ensure_loaded(["docs"])
    console.print("\n[bold cyan][ Docs Agent — Summarize ][/bold cyan]")

    prompt = f"Summarize this clearly and concisely:\n\n{content}"

    if streaming:
        result = ""
        for chunk in stream(prompt, role="docs", system_prompt=SUMMARY_SYSTEM):
            result += chunk
    else:
        result = ask(prompt, role="docs", system_prompt=SUMMARY_SYSTEM)
        console.print(result)

    save_task("docs", prompt, result, "docs")
    return result