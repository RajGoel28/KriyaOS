# ============================================================
#  KriyaOS — orchestrator/classifier.py
#  Intent classifier — figures out what the user wants.
#  Uses all-MiniLM-L6-v2 (sentence-transformers) for fast,
#  lightweight classification. Runs on CPU, no LM Studio needed.
#  Fine-tune this on Kaggle with CLINC150 dataset later.
# ============================================================

from sentence_transformers import SentenceTransformer, util
from rich.console import Console
import torch

console = Console()

# ============================================================
#  TASK CATEGORIES
#  These are the intents KriyaOS understands.
#  Each category maps to a specific agent in the orchestrator.
# ============================================================

CATEGORIES: dict[str, list[str]] = {

    "code": [
        "write a function",
        "generate code",
        "create a script",
        "write a program",
        "implement this feature",
        "code this for me",
        "write a class",
        "make a module",
        "build an API",
        "write a decorator",
    ],

    "fix": [
        "fix this bug",
        "debug this code",
        "there is an error",
        "why is this not working",
        "exception in my code",
        "traceback error",
        "fix the issue",
        "something is broken",
        "error in my script",
        "my code crashes",
    ],

    "explain": [
        "explain this code",
        "what does this do",
        "how does this work",
        "break this down",
        "walk me through",
        "what is this function",
        "help me understand",
        "describe this",
        "what does this mean",
        "clarify this",
    ],

    "build": [
        "build a full project",
        "create an application",
        "scaffold a new app",
        "set up a project",
        "build a web app",
        "create a REST API",
        "make a full stack app",
        "generate a project structure",
        "build a CLI tool",
        "create a complete system",
    ],

    "plan": [
        "plan this project",
        "create a roadmap",
        "how should I approach this",
        "what steps should I take",
        "design the architecture",
        "break this into tasks",
        "help me structure this",
        "what is the best approach",
        "outline the steps",
        "help me think through this",
    ],

    "review": [
        "review this code",
        "check my code",
        "is this code good",
        "any improvements",
        "critique this",
        "what can be improved",
        "give feedback on this",
        "is this correct",
        "review my implementation",
        "check for issues",
    ],

    "docs": [
        "write documentation",
        "create a README",
        "write docstrings",
        "document this code",
        "create API docs",
        "write comments",
        "generate docs",
        "write a technical guide",
        "create a wiki page",
        "write a tutorial",
    ],

    "design": [
        "design a UI",
        "create a mockup",
        "design the interface",
        "draw a wireframe",
        "generate a layout",
        "design a screen",
        "create a component",
        "design a dashboard",
        "make a UI design",
        "create a frontend",
    ],

    "ask": [
        "what is",
        "who is",
        "tell me about",
        "how do I",
        "give me information",
        "I have a question",
        "can you help me with",
        "what are the best",
        "recommend me",
        "which is better",
    ],

    "voice": [
        "listen to me",
        "voice input",
        "speech mode",
        "talk to me",
        "use microphone",
        "voice command",
        "activate voice",
        "start listening",
        "speech recognition",
        "speak to kriya",
    ],
}

# ============================================================
#  MODEL SETUP
# ============================================================

_model: SentenceTransformer = None
_category_embeddings: dict[str, torch.Tensor] = {}


def _load_model() -> None:
    """
    Load all-MiniLM-L6-v2 and pre-compute category embeddings.
    Called once on first use — subsequent calls are instant.
    """
    global _model, _category_embeddings

    if _model is not None:
        return  # already loaded

    console.print("[dim][ classifier ] Loading all-MiniLM-L6-v2...[/dim]")
    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Pre-compute embeddings for all category examples
    for category, examples in CATEGORIES.items():
        embeddings = _model.encode(examples, convert_to_tensor=True)
        # Use mean of all example embeddings as the category centroid
        _category_embeddings[category] = embeddings.mean(dim=0)

    console.print("[dim][ classifier ] Ready.[/dim]")


# ============================================================
#  CLASSIFY
# ============================================================

def classify(prompt: str, top_k: int = 1) -> list[dict]:
    """
    Classify a prompt into one or more KriyaOS task categories.
    Uses cosine similarity between the prompt embedding and
    pre-computed category centroids.

    Args:
        prompt: The user's input text
        top_k:  How many top categories to return (default: 1)

    Returns:
        List of dicts sorted by confidence score (highest first):
        [{"category": "code", "score": 0.91}, ...]

    Usage:
        result = classify("Write a FastAPI server")
        print(result[0]["category"])  # "code"

        # Get top 3 categories
        results = classify("Build and document a REST API", top_k=3)
    """
    _load_model()

    prompt_embedding = _model.encode(prompt, convert_to_tensor=True)

    scores = []
    for category, centroid in _category_embeddings.items():
        score = util.cos_sim(prompt_embedding, centroid).item()
        scores.append({"category": category, "score": round(score, 4)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


def get_category(prompt: str) -> str:
    """
    Get the single best category for a prompt.
    This is what the router calls to decide which agent to use.

    Args:
        prompt: The user's input text

    Returns:
        Category string e.g. "code", "fix", "plan", "ask"

    Usage:
        category = get_category("Fix this bug in my code")
        # returns "fix"
    """
    return classify(prompt, top_k=1)[0]["category"]


def get_confidence(prompt: str) -> float:
    """
    Get the confidence score for the top category.
    Score is between 0.0 and 1.0.
    Below 0.3 means the prompt is ambiguous.

    Usage:
        score = get_confidence("Write a sort function")
        if score < 0.3:
            print("Low confidence — ask user for clarification")
    """
    return classify(prompt, top_k=1)[0]["score"]


# ============================================================
#  CATEGORY → AGENT MAPPING
#  Tells the router which agent handles each category
# ============================================================

CATEGORY_TO_AGENT: dict[str, str] = {
    "code":    "backend_agent",
    "fix":     "backend_agent",
    "explain": "docs_agent",
    "build":   "planner",
    "plan":    "planner",
    "review":  "reviewer",
    "docs":    "docs_agent",
    "design":  "design_agent",
    "ask":     "router",       # simple ask goes to router model directly
    "voice":   "router",       # voice handled separately
}


def get_agent(prompt: str) -> str:
    """
    Get the agent name for a given prompt.

    Usage:
        agent = get_agent("Write a FastAPI server")
        # returns "backend_agent"
    """
    category = get_category(prompt)
    return CATEGORY_TO_AGENT.get(category, "router")


# ============================================================
#  QUICK TEST — python orchestrator/classifier.py
# ============================================================

if __name__ == "__main__":
    from rich.table import Table

    console.rule("[bold cyan]KriyaOS — classifier test[/bold cyan]")

    test_prompts = [
        "Write a Python function to sort a list",
        "Fix this bug: TypeError on line 42",
        "Explain what this decorator does",
        "Build a complete REST API with auth",
        "Plan out my KriyaOS project architecture",
        "Review my FastAPI code for issues",
        "Write documentation for this module",
        "Design a dashboard UI for KriyaOS",
        "What is the difference between async and sync?",
        "Activate voice input mode",
    ]

    table = Table(title="Classifier Results", show_lines=True)
    table.add_column("Prompt",     style="white",   max_width=42)
    table.add_column("Category",   style="cyan",    no_wrap=True)
    table.add_column("Agent",      style="magenta", no_wrap=True)
    table.add_column("Confidence", style="yellow",  justify="right")

    for prompt in test_prompts:
        results  = classify(prompt, top_k=1)
        category = results[0]["category"]
        score    = results[0]["score"]
        agent    = CATEGORY_TO_AGENT.get(category, "router")
        conf_str = f"{score:.2f}" + (" ✓" if score >= 0.4 else " ?")

        table.add_row(
            prompt[:42],
            category,
            agent,
            conf_str,
        )

    console.print(table)
    console.rule("[bold green]Classifier test done![/bold green]")