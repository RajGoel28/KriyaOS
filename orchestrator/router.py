# ============================================================
#  KriyaOS — orchestrator/router.py
#  The decision maker. Combines classifier + complexity scorer
#  and returns a full routing decision for every task.
#
#  Flow:
#    user prompt
#      → classifier  (what is this? e.g. "code")
#      → complexity  (how hard? e.g. "complex")
#      → router      (which agent + models + pipeline?)
#      → agent runs the task
# ============================================================

import time
from dataclasses import dataclass
from rich.console import Console

from classifier import get_category, get_confidence, classify, CATEGORY_TO_AGENT
from complexity import score as complexity_score, ComplexityResult, Level

console = Console()


# ============================================================
#  ROUTING DECISION
#  Everything the orchestrator needs to run a task
# ============================================================

@dataclass
class RoutingDecision:
    """
    Full routing decision returned by the router.
    The orchestrator reads this and runs the task accordingly.
    """
    # What the task is
    prompt:       str
    category:     str        # "code", "fix", "plan", etc.
    confidence:   float      # classifier confidence 0.0-1.0

    # How hard it is
    complexity:   ComplexityResult

    # What to run
    agent:        str        # which agent handles it
    models:       list[str]  # which model roles to load
    use_ensemble: bool       # run debate engine?
    pipeline:     str        # "direct" / "single" / "ensemble" / "full_team"

    # Meta
    routing_time_ms: int     # how long routing took


# ============================================================
#  PIPELINE TYPES
#  Tells the orchestrator HOW to run the task
# ============================================================

class Pipeline:
    DIRECT     = "direct"      # router model answers directly (simple/fast)
    SINGLE     = "single"      # one specialist agent
    ENSEMBLE   = "ensemble"    # generate → critique → polish (2 models)
    FULL_TEAM  = "full_team"   # planner + agents + reviewer + ensemble


# ============================================================
#  PIPELINE SELECTION
#  Based on complexity level + category
# ============================================================

def _select_pipeline(level: str, category: str) -> str:
    """
    Pick the right pipeline for this task.

    simple  → direct (router model answers, no agent needed)
    medium  → single (one specialist agent)
    complex → ensemble (two models debate)
    expert  → full_team (all agents involved)

    Exception: "ask" category always uses direct even if medium.
    """
    if category == "ask" or level == Level.SIMPLE:
        return Pipeline.DIRECT

    if level == Level.MEDIUM:
        return Pipeline.SINGLE

    if level == Level.COMPLEX:
        return Pipeline.ENSEMBLE

    # expert
    return Pipeline.FULL_TEAM


# ============================================================
#  CONFIDENCE FALLBACK
#  If classifier is not confident, use a safer default
# ============================================================

CONFIDENCE_THRESHOLD = 0.35  # below this → fall back to "ask" category

def _apply_confidence_fallback(category: str, confidence: float) -> str:
    """
    If classifier confidence is too low, fall back to generic "ask"
    rather than routing to the wrong specialist agent.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        console.print(
            f"[dim][ router ] Low confidence ({confidence:.2f}) for '{category}' "
            f"— falling back to 'ask'[/dim]"
        )
        return "ask"
    return category


# ============================================================
#  MAIN ROUTER
# ============================================================

def route(prompt: str) -> RoutingDecision:
    """
    Main routing function. Takes a raw user prompt and returns
    a complete RoutingDecision the orchestrator can act on.

    Args:
        prompt: The user's raw input text

    Returns:
        RoutingDecision with everything needed to run the task

    Usage:
        decision = route("Write a FastAPI server with JWT auth")
        print(decision.category)       # "code"
        print(decision.complexity.level)  # "complex"
        print(decision.pipeline)       # "ensemble"
        print(decision.models)         # ["coder", "orchestrator"]
        print(decision.agent)          # "backend_agent"
    """
    start = time.time()

    # 1. Classify intent
    results    = classify(prompt, top_k=1)
    category   = results[0]["category"]
    confidence = results[0]["score"]

    # 2. Apply confidence fallback
    category = _apply_confidence_fallback(category, confidence)

    # 3. Score complexity
    complexity = complexity_score(prompt, category)

    # 4. Select agent
    agent = CATEGORY_TO_AGENT.get(category, "router")

    # 5. Select pipeline
    pipeline = _select_pipeline(complexity.level, category)

    # 6. Override models for full_team pipeline
    models = complexity.recommended_models
    if pipeline == Pipeline.FULL_TEAM:
        models = ["planner", "coder", "orchestrator", "reviewer"]

    routing_ms = int((time.time() - start) * 1000)

    return RoutingDecision(
        prompt          = prompt,
        category        = category,
        confidence      = confidence,
        complexity      = complexity,
        agent           = agent,
        models          = models,
        use_ensemble    = complexity.use_ensemble,
        pipeline        = pipeline,
        routing_time_ms = routing_ms,
    )


def route_batch(prompts: list[str]) -> list[RoutingDecision]:
    """
    Route multiple prompts at once.

    Usage:
        decisions = route_batch(["Fix my bug", "Build a full app"])
        for d in decisions:
            print(d.category, d.pipeline)
    """
    return [route(p) for p in prompts]


# ============================================================
#  DISPLAY HELPER
# ============================================================

def print_decision(decision: RoutingDecision) -> None:
    """
    Pretty print a routing decision to the terminal.
    Used by CLI commands to show what KriyaOS is doing.

    Usage:
        decision = route("Build a REST API")
        print_decision(decision)
    """
    LEVEL_COLORS = {
        Level.SIMPLE:  "green",
        Level.MEDIUM:  "yellow",
        Level.COMPLEX: "orange3",
        Level.EXPERT:  "red",
    }

    PIPELINE_COLORS = {
        Pipeline.DIRECT:    "green",
        Pipeline.SINGLE:    "yellow",
        Pipeline.ENSEMBLE:  "orange3",
        Pipeline.FULL_TEAM: "red",
    }

    level_color    = LEVEL_COLORS.get(decision.complexity.level, "white")
    pipeline_color = PIPELINE_COLORS.get(decision.pipeline, "white")

    console.print(f"\n[bold cyan][ KriyaOS Router ][/bold cyan]")
    console.print(f"  Prompt     : {decision.prompt[:72]}")
    console.print(f"  Category   : [cyan]{decision.category}[/cyan]  (confidence: {decision.confidence:.2f})")
    console.print(f"  Complexity : [{level_color}]{decision.complexity.level}[/{level_color}]  (score: {decision.complexity.score})")
    console.print(f"  Pipeline   : [{pipeline_color}]{decision.pipeline}[/{pipeline_color}]")
    console.print(f"  Agent      : [magenta]{decision.agent}[/magenta]")
    console.print(f"  Models     : [green]{', '.join(decision.models)}[/green]")
    console.print(f"  Ensemble   : {'yes' if decision.use_ensemble else 'no'}")
    console.print(f"  ETA        : {decision.complexity.estimated_time}")
    console.print(f"  Routed in  : {decision.routing_time_ms}ms\n")


# ============================================================
#  QUICK TEST — python orchestrator/router.py
# ============================================================

if __name__ == "__main__":
    from rich.table import Table

    console.rule("[bold cyan]KriyaOS — router test[/bold cyan]")

    test_prompts = [
        "What is a decorator in Python?",
        "Fix this bug: IndexError on line 12",
        "Write a binary search function",
        "Review my authentication code",
        "Build a complete e-commerce API with auth, payments, and database",
        "Plan the full KriyaOS orchestrator architecture",
        "Design a dashboard UI for KriyaOS",
        "Write documentation for this FastAPI module",
        "Build a scalable distributed microservice system with caching, queues, and deploy it to production",
        "Explain what async/await does",
    ]

    # Summary table
    table = Table(title="Router Decisions", show_lines=True)
    table.add_column("Prompt",     style="white",   max_width=36)
    table.add_column("Category",   style="cyan",    no_wrap=True)
    table.add_column("Level",      style="yellow",  no_wrap=True)
    table.add_column("Pipeline",   style="magenta", no_wrap=True)
    table.add_column("Agent",      style="green",   no_wrap=True)
    table.add_column("ms",         style="dim",     justify="right")

    for prompt in test_prompts:
        d = route(prompt)
        table.add_row(
            prompt[:36],
            d.category,
            d.complexity.level,
            d.pipeline,
            d.agent,
            str(d.routing_time_ms),
        )

    console.print(table)

    # Detailed view of one complex case
    console.print("\n[bold]Detailed decision — complex case:[/bold]")
    decision = route("Build a complete e-commerce API with auth, payments, and database")
    print_decision(decision)

    console.rule("[bold green]Router test done![/bold green]")