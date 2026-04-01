# ============================================================
#  KriyaOS — orchestrator/complexity.py
#  Rates how complex a task is: simple / medium / complex / expert
#  This decides HOW MANY agents to use and WHICH models to load.
#
#  simple  → 1 model, direct answer, < 15 seconds
#  medium  → 1-2 models, some reasoning, < 60 seconds
#  complex → 2 models + ensemble, < 5 minutes
#  expert  → full agent team + debate, < 15 minutes
# ============================================================

import re
from dataclasses import dataclass
from rich.console import Console

console = Console()


# ============================================================
#  COMPLEXITY LEVELS
# ============================================================

class Level:
    SIMPLE  = "simple"
    MEDIUM  = "medium"
    COMPLEX = "complex"
    EXPERT  = "expert"


@dataclass
class ComplexityResult:
    """
    Full result from the complexity scorer.
    """
    level:           str    # simple / medium / complex / expert
    score:           int    # raw score 0-100
    reasons:         list[str]  # why this score was given
    recommended_models: list[str]  # which model roles to use
    use_ensemble:    bool   # whether to run the debate engine
    estimated_time:  str    # rough time estimate


# ============================================================
#  SCORING SIGNALS
#  Each signal adds points to the complexity score.
#  Score thresholds → level:
#    0-24   = simple
#    25-49  = medium
#    50-74  = complex
#    75-100 = expert
# ============================================================

# Words/phrases that increase complexity
COMPLEX_SIGNALS: list[tuple[str, int]] = [
    # Architecture / system design
    (r"\barchitect\b",              15),
    (r"\bsystem design\b",          15),
    (r"\bscalable\b",               10),
    (r"\bmicroservice",             15),
    (r"\bdistributed\b",            15),
    (r"\bfull.?stack\b",            12),
    (r"\bend.to.end\b",             12),

    # Multi-step / full project
    (r"\bcomplete\b",               10),
    (r"\bfrom scratch\b",           12),
    (r"\bfull project\b",           15),
    (r"\bwhole (app|system)\b",     15),
    (r"\bproduction.ready\b",       15),
    (r"\bdeploy\b",                 10),

    # Multiple components
    (r"\bauth(entication)?\b",      8),
    (r"\bdatabase\b",               6),
    (r"\bapi\b",                    4),
    (r"\bfrontend\b",               6),
    (r"\bbackend\b",                6),
    (r"\bcaching\b",                8),
    (r"\bqueue\b",                  8),
    (r"\bwebsocket\b",              8),

    # Advanced concepts
    (r"\bmachine learning\b",       15),
    (r"\bneural network\b",         15),
    (r"\boptimize\b",               8),
    (r"\bperformance\b",            6),
    (r"\bconcurren(t|cy)\b",        10),
    (r"\basync\b",                  5),
    (r"\bmulti.?thread",            10),
    (r"\bsecurity\b",               8),
    (r"\bencrypt\b",                8),

    # Quantity signals
    (r"\bmultiple\b",               6),
    (r"\bseveral\b",                5),
    (r"\bmany\b",                   4),
    (r"\ball\b",                    4),

    # Planning / design
    (r"\bplan\b",                   5),
    (r"\bdesign\b",                 5),
    (r"\barchitecture\b",           12),
    (r"\broadmap\b",                10),
    (r"\bstrategy\b",               8),
]

# Words that REDUCE complexity
SIMPLE_SIGNALS: list[tuple[str, int]] = [
    (r"\bwhat is\b",               -10),
    (r"\bwho is\b",                -10),
    (r"\bdefine\b",                -8),
    (r"\bexample\b",               -5),
    (r"\bsimple\b",                -8),
    (r"\bquick\b",                 -8),
    (r"\bbasic\b",                 -8),
    (r"\bone.?liner\b",            -10),
    (r"\bjust\b",                  -5),
    (r"\bonly\b",                  -5),
    (r"\bsmall\b",                 -5),
    (r"\bshort\b",                 -5),
]


# ============================================================
#  LENGTH SCORING
#  Longer prompts usually = more complex tasks
# ============================================================

def _length_score(prompt: str) -> tuple[int, str]:
    words = len(prompt.split())
    if words < 8:
        return 0, f"short prompt ({words} words)"
    elif words < 20:
        return 5, f"medium prompt ({words} words)"
    elif words < 50:
        return 10, f"detailed prompt ({words} words)"
    else:
        return 20, f"very detailed prompt ({words} words)"


# ============================================================
#  LINE / CODE BLOCK SCORING
#  If prompt contains code, it's likely a fix/review task
# ============================================================

def _code_score(prompt: str) -> tuple[int, str]:
    lines = prompt.count("\n")
    has_code_block = "```" in prompt or "def " in prompt or "class " in prompt
    if has_code_block and lines > 20:
        return 15, "large code block included"
    elif has_code_block:
        return 8, "code block included"
    elif lines > 5:
        return 5, "multi-line prompt"
    return 0, ""


# ============================================================
#  CATEGORY BASELINE SCORES
#  Some task types are inherently more complex than others
# ============================================================

CATEGORY_BASELINE: dict[str, int] = {
    "ask":     0,
    "explain": 5,
    "fix":     10,
    "review":  10,
    "docs":    10,
    "code":    15,
    "design":  15,
    "plan":    20,
    "build":   30,
    "voice":   0,
}


# ============================================================
#  MODEL RECOMMENDATIONS PER LEVEL
#  Which model roles to load for each complexity level
# ============================================================

MODEL_RECOMMENDATIONS: dict[str, list[str]] = {
    Level.SIMPLE:  ["router"],
    Level.MEDIUM:  ["router", "coder"],
    Level.COMPLEX: ["coder", "orchestrator"],
    Level.EXPERT:  ["planner", "coder", "orchestrator", "reviewer"],
}


# ============================================================
#  MAIN SCORER
# ============================================================

def score(prompt: str, category: str = "ask") -> ComplexityResult:
    """
    Score a prompt's complexity and return a full ComplexityResult.

    Args:
        prompt:   The user's input text
        category: The classified category from classifier.py

    Returns:
        ComplexityResult with level, score, reasons, model recommendations

    Usage:
        from orchestrator.classifier import get_category
        from orchestrator.complexity import score

        category = get_category(prompt)
        result   = score(prompt, category)
        print(result.level)           # "complex"
        print(result.recommended_models)  # ["coder", "orchestrator"]
        print(result.use_ensemble)    # True
    """
    total  = 0
    reasons = []

    # 1. Category baseline
    baseline = CATEGORY_BASELINE.get(category, 0)
    if baseline > 0:
        total += baseline
        reasons.append(f"category '{category}' baseline +{baseline}")

    # 2. Length score
    length_pts, length_reason = _length_score(prompt)
    if length_pts > 0:
        total += length_pts
        reasons.append(length_reason + f" +{length_pts}")

    # 3. Code block score
    code_pts, code_reason = _code_score(prompt)
    if code_pts > 0:
        total += code_pts
        reasons.append(code_reason + f" +{code_pts}")

    # 4. Signal matching
    prompt_lower = prompt.lower()
    for pattern, points in COMPLEX_SIGNALS:
        if re.search(pattern, prompt_lower):
            word = pattern.replace(r"\b", "").replace("\\b", "").strip("()?")
            total += points
            reasons.append(f"signal '{word}' +{points}")

    for pattern, points in SIMPLE_SIGNALS:
        if re.search(pattern, prompt_lower):
            word = pattern.replace(r"\b", "").replace("\\b", "").strip("()?")
            total += points  # points are negative
            reasons.append(f"simplifier '{word}' {points}")

    # Clamp to 0-100
    total = max(0, min(100, total))

    # 5. Determine level
    if total < 25:
        level = Level.SIMPLE
        estimated_time = "< 15 seconds"
        use_ensemble   = False
    elif total < 50:
        level = Level.MEDIUM
        estimated_time = "< 60 seconds"
        use_ensemble   = False
    elif total < 75:
        level = Level.COMPLEX
        estimated_time = "< 5 minutes"
        use_ensemble   = True
    else:
        level = Level.EXPERT
        estimated_time = "< 15 minutes"
        use_ensemble   = True

    recommended_models = MODEL_RECOMMENDATIONS[level]

    return ComplexityResult(
        level            = level,
        score            = total,
        reasons          = reasons,
        recommended_models = recommended_models,
        use_ensemble     = use_ensemble,
        estimated_time   = estimated_time,
    )


def get_level(prompt: str, category: str = "ask") -> str:
    """
    Quick helper — just returns the level string.

    Usage:
        level = get_level("Write a sort function", "code")
        # "simple"
    """
    return score(prompt, category).level


# ============================================================
#  QUICK TEST — python orchestrator/complexity.py
# ============================================================

if __name__ == "__main__":
    from rich.table import Table

    console.rule("[bold cyan]KriyaOS — complexity test[/bold cyan]")

    test_cases = [
        ("What is a Python list?",                                "ask"),
        ("Fix this bug: TypeError on line 42",                    "fix"),
        ("Write a function to reverse a string",                  "code"),
        ("Review my FastAPI authentication code",                 "review"),
        ("Build a complete REST API with auth and database",      "build"),
        ("Design and implement a scalable microservice system",   "build"),
        ("Write a full stack app with React frontend, FastAPI backend, PostgreSQL, Redis caching, JWT auth, and deploy it", "build"),
        ("Plan the full KriyaOS architecture from scratch",       "plan"),
        ("What is async?",                                        "ask"),
        ("Implement a distributed task queue with concurrency",   "code"),
    ]

    table = Table(title="Complexity Results", show_lines=True)
    table.add_column("Prompt",     style="white",   max_width=38)
    table.add_column("Level",      style="cyan",    no_wrap=True)
    table.add_column("Score",      style="yellow",  justify="right")
    table.add_column("Ensemble",   style="magenta", justify="center")
    table.add_column("Models",     style="green",   max_width=30)
    table.add_column("ETA",        style="dim",     no_wrap=True)

    LEVEL_COLORS = {
        Level.SIMPLE:  "green",
        Level.MEDIUM:  "yellow",
        Level.COMPLEX: "orange3",
        Level.EXPERT:  "red",
    }

    for prompt, category in test_cases:
        result = score(prompt, category)
        color  = LEVEL_COLORS[result.level]
        table.add_row(
            prompt[:38],
            f"[{color}]{result.level}[/{color}]",
            str(result.score),
            "yes" if result.use_ensemble else "no",
            ", ".join(result.recommended_models),
            result.estimated_time,
        )

    console.print(table)

    # Show detailed breakdown for one case
    console.print("\n[bold]Detailed breakdown — full stack prompt:[/bold]")
    detailed = score(
        "Write a full stack app with React frontend, FastAPI backend, PostgreSQL, Redis caching, JWT auth, and deploy it",
        "build"
    )
    for reason in detailed.reasons:
        console.print(f"  [dim]+[/dim] {reason}")
    console.print(f"\n  [bold]Final score:[/bold] {detailed.score} → [red]{detailed.level}[/red]")

    console.rule("[bold green]Complexity test done![/bold green]")