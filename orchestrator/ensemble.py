# ============================================================
#  KriyaOS — orchestrator/ensemble.py
#  The debate engine. Makes KriyaOS smarter than any single model.
#
#  3-step pipeline:
#    Step 1 — GENERATE  : specialist model writes first answer
#    Step 2 — CRITIQUE  : critic model finds flaws and gaps
#    Step 3 — POLISH    : specialist rewrites using the critique
#
#  Used for complex + expert tasks automatically by the router.
#  Result is always better than what any single model produces.
# ============================================================

import time
import sys
import os
from dataclasses import dataclass
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Allow running from orchestrator/ directory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ai_core import ask, stream
from core.memory import save_task
from orchestrator.bus import bus, Topic

console = Console()


# ============================================================
#  ENSEMBLE RESULT
# ============================================================

@dataclass
class EnsembleResult:
    """
    Full result from one ensemble run.
    Contains all 3 steps so you can inspect the full pipeline.
    """
    prompt:        str
    draft:         str        # Step 1 — first model output
    critique:      str        # Step 2 — critic's feedback
    final:         str        # Step 3 — polished final answer
    generator_role: str       # which model generated
    critic_role:    str       # which model critiqued
    total_time_s:  float      # total time for all 3 steps
    step_times:    dict       # time per step in seconds


# ============================================================
#  SYSTEM PROMPTS
# ============================================================

GENERATOR_SYSTEM = """You are Kriya, an expert AI assistant inside KriyaOS.
Your job is to produce the best possible answer to the user's request.
Be thorough, correct, and well-structured. Use code examples where relevant.
Do not hold back — give your complete best answer."""

CRITIC_SYSTEM = """You are a strict technical critic inside KriyaOS.
You will be given a prompt and a draft answer.
Your job is to find EVERY flaw, gap, error, and improvement opportunity.
Be ruthless but constructive. Structure your critique as:

ISSUES:
- [list every problem you find]

MISSING:
- [list anything important that was left out]

IMPROVEMENTS:
- [specific suggestions to make it better]

Do not rewrite the answer — only critique it."""

POLISHER_SYSTEM = """You are Kriya, an expert AI assistant inside KriyaOS.
You will be given:
1. The original user prompt
2. A draft answer
3. A critique of that draft

Your job is to produce a FINAL, POLISHED answer that:
- Fixes every issue mentioned in the critique
- Adds everything listed as missing
- Implements all suggested improvements
- Is better than the draft in every way

Produce only the final answer — no meta-commentary."""


# ============================================================
#  STEP RUNNERS
# ============================================================

def _run_generate(
    prompt:         str,
    generator_role: str,
    streaming:      bool,
) -> tuple[str, float]:
    """
    Step 1 — Generate the first draft.
    Returns (draft_text, time_taken_seconds)
    """
    console.print(Rule("[cyan]Step 1 — Generating draft[/cyan]"))
    console.print(f"[dim]Model: {generator_role}[/dim]\n")

    start = time.time()

    if streaming:
        draft = ""
        for chunk in stream(prompt, role=generator_role, system_prompt=GENERATOR_SYSTEM):
            draft += chunk
    else:
        draft = ask(prompt, role=generator_role, system_prompt=GENERATOR_SYSTEM)
        console.print(draft)

    elapsed = round(time.time() - start, 2)
    console.print(f"\n[dim]Draft complete in {elapsed}s[/dim]")

    # Post draft to bus
    bus.post(Topic.ENSEMBLE_DRAFT, generator_role, {
        "prompt": prompt,
        "draft":  draft,
    })

    return draft, elapsed


def _run_critique(
    prompt:      str,
    draft:       str,
    critic_role: str,
    streaming:   bool,
) -> tuple[str, float]:
    """
    Step 2 — Critique the draft.
    Returns (critique_text, time_taken_seconds)
    """
    console.print(Rule("[yellow]Step 2 — Critiquing draft[/yellow]"))
    console.print(f"[dim]Model: {critic_role}[/dim]\n")

    critique_prompt = (
        f"ORIGINAL REQUEST:\n{prompt}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        f"Critique this draft thoroughly."
    )

    start = time.time()

    if streaming:
        critique = ""
        for chunk in stream(
            critique_prompt,
            role=critic_role,
            system_prompt=CRITIC_SYSTEM,
        ):
            critique += chunk
    else:
        critique = ask(
            critique_prompt,
            role=critic_role,
            system_prompt=CRITIC_SYSTEM,
        )
        console.print(critique)

    elapsed = round(time.time() - start, 2)
    console.print(f"\n[dim]Critique complete in {elapsed}s[/dim]")

    # Post critique to bus
    bus.post(Topic.ENSEMBLE_CRITIQUE, critic_role, {
        "prompt":   prompt,
        "draft":    draft,
        "critique": critique,
    })

    return critique, elapsed


def _run_polish(
    prompt:         str,
    draft:          str,
    critique:       str,
    generator_role: str,
    streaming:      bool,
) -> tuple[str, float]:
    """
    Step 3 — Polish the draft using the critique.
    Returns (final_text, time_taken_seconds)
    """
    console.print(Rule("[green]Step 3 — Polishing final answer[/green]"))
    console.print(f"[dim]Model: {generator_role}[/dim]\n")

    polish_prompt = (
        f"ORIGINAL REQUEST:\n{prompt}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        f"CRITIQUE:\n{critique}\n\n"
        f"Now produce the final, polished answer."
    )

    start = time.time()

    if streaming:
        final = ""
        for chunk in stream(
            polish_prompt,
            role=generator_role,
            system_prompt=POLISHER_SYSTEM,
        ):
            final += chunk
    else:
        final = ask(
            polish_prompt,
            role=generator_role,
            system_prompt=POLISHER_SYSTEM,
        )
        console.print(final)

    elapsed = round(time.time() - start, 2)
    console.print(f"\n[dim]Polish complete in {elapsed}s[/dim]")

    # Post final to bus
    bus.post(Topic.ENSEMBLE_FINAL, generator_role, {
        "prompt": prompt,
        "final":  final,
    })

    return final, elapsed


# ============================================================
#  MAIN ENSEMBLE RUNNER
# ============================================================

def run(
    prompt:         str,
    generator_role: str = "coder",
    critic_role:    str = "orchestrator",
    streaming:      bool = True,
    save_to_memory: bool = True,
    task_type:      str = "ensemble",
) -> EnsembleResult:
    """
    Run the full 3-step ensemble pipeline.

    Args:
        prompt:         The user's task/question
        generator_role: Model that generates + polishes (default: coder)
        critic_role:    Model that critiques (default: orchestrator)
        streaming:      Stream output live to terminal
        save_to_memory: Auto-save result to SQLite training data
        task_type:      Task category for memory logging

    Returns:
        EnsembleResult with draft, critique, final, and timing

    Usage:
        from orchestrator.ensemble import run

        result = run(
            "Write a FastAPI server with JWT authentication",
            generator_role="coder",
            critic_role="orchestrator",
        )
        print(result.final)
        print(f"Total time: {result.total_time_s}s")
    """
    total_start = time.time()

    console.print(Panel(
        f"[bold cyan]KriyaOS Ensemble Engine[/bold cyan]\n"
        f"[dim]Generator: {generator_role}  |  Critic: {critic_role}[/dim]\n"
        f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/dim]",
        expand=False,
    ))

    # Step 1 — Generate
    draft, t1 = _run_generate(prompt, generator_role, streaming)

    # Step 2 — Critique
    critique, t2 = _run_critique(prompt, draft, critic_role, streaming)

    # Step 3 — Polish
    final, t3 = _run_polish(prompt, draft, critique, generator_role, streaming)

    total_time = round(time.time() - total_start, 2)

    # Show summary
    console.print(Panel(
        f"[bold green]Ensemble complete[/bold green]\n"
        f"  Step 1 generate : {t1}s\n"
        f"  Step 2 critique : {t2}s\n"
        f"  Step 3 polish   : {t3}s\n"
        f"  Total           : {total_time}s",
        expand=False,
    ))

    result = EnsembleResult(
        prompt         = prompt,
        draft          = draft,
        critique       = critique,
        final          = final,
        generator_role = generator_role,
        critic_role    = critic_role,
        total_time_s   = total_time,
        step_times     = {"generate": t1, "critique": t2, "polish": t3},
    )

    # Auto-save to memory for fine-tuning dataset
    if save_to_memory:
        save_task(
            task_type  = task_type,
            prompt     = prompt,
            response   = final,
            model_role = generator_role,
            duration_ms = int(total_time * 1000),
        )

    return result


# ============================================================
#  QUICK ENSEMBLE — skip critique, just generate + polish
#  Use for medium tasks that need improvement but not full debate
# ============================================================

def run_quick(
    prompt:         str,
    generator_role: str = "router",
    streaming:      bool = True,
) -> str:
    """
    Lightweight 2-step pipeline: generate → self-critique → polish.
    Same model does all 3 steps. Faster, uses only 1 model.
    Good for medium complexity tasks.

    Returns:
        Final polished answer as string

    Usage:
        answer = run_quick("Explain Python decorators", role="router")
    """
    console.print(Rule("[cyan]Quick Ensemble — single model[/cyan]"))

    # Step 1 — Generate
    if streaming:
        draft = ""
        for chunk in stream(prompt, role=generator_role, system_prompt=GENERATOR_SYSTEM):
            draft += chunk
    else:
        draft = ask(prompt, role=generator_role, system_prompt=GENERATOR_SYSTEM)

    # Step 2 — Self polish (same model critiques and rewrites itself)
    polish_prompt = (
        f"Original request: {prompt}\n\n"
        f"Your first answer: {draft}\n\n"
        f"Now improve this answer. Fix any issues, add missing details, "
        f"and make it clearer and more complete."
    )

    console.print(Rule("[green]Polishing...[/green]"))

    if streaming:
        final = ""
        for chunk in stream(polish_prompt, role=generator_role, system_prompt=POLISHER_SYSTEM):
            final += chunk
    else:
        final = ask(polish_prompt, role=generator_role, system_prompt=POLISHER_SYSTEM)

    return final


# ============================================================
#  QUICK TEST — python orchestrator/ensemble.py
#  Make sure LM Studio is running with at least one model loaded
# ============================================================

if __name__ == "__main__":
    from core.ai_core import is_online

    console.rule("[bold cyan]KriyaOS — ensemble test[/bold cyan]")

    if not is_online():
        console.print("[red]LM Studio is not running.[/red]")
        console.print("Start LM Studio → load a model → Local Server → Start Server")
        exit(1)

    console.print("[green]LM Studio online.[/green]\n")

    # Test with one model doing both roles (since you may only have Phi-4-mini loaded)
    # In production: generator_role="coder", critic_role="orchestrator"
    test_prompt = "Write a Python function that checks if a string is a palindrome"

    console.print("[dim]Note: Using 'router' for both roles since only one model may be loaded.[/dim]")
    console.print("[dim]In production, generator=coder and critic=orchestrator.[/dim]\n")

    result = run(
        prompt         = test_prompt,
        generator_role = "router",
        critic_role    = "router",   # same model critiques itself
        streaming      = True,
        save_to_memory = True,
        task_type      = "code",
    )

    console.print("\n[bold]Final answer length:[/bold]", len(result.final), "chars")
    console.print("[bold]Step times:[/bold]", result.step_times)
    console.print("[bold]Total time:[/bold]", result.total_time_s, "s")

    console.rule("[bold green]Ensemble test done![/bold green]")