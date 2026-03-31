# ============================================================
#  KriyaOS — core/model_registry.py
#  Single source of truth for all models, roles, and settings.
#  Every other file imports from here — never hardcode model names.
# ============================================================

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for a single model in the KriyaOS registry.
    """
    name: str                          # Display name
    model_id: str                      # Exact ID used in LM Studio API
    role: str                          # What this model does in KriyaOS
    size_gb: float                     # Approximate size on disk (Q4)
    context_length: int                # Max tokens this model supports
    temperature: float = 0.7           # Default temperature
    max_tokens: int = 2048             # Default max response tokens
    is_finetuned: bool = False         # True once you fine-tune it on Kaggle
    adapter_path: Optional[str] = None # Path to LoRA adapter if fine-tuned
    notes: str = ""                    # Any extra notes


# ============================================================
#  ALL KRIYAOS MODELS
#  Install in this exact order (smallest first saves RAM)
# ============================================================

MODELS: dict[str, ModelConfig] = {

    # --- Classifier (sentence-transformers, not LM Studio) ---
    "classifier": ModelConfig(
        name        = "all-MiniLM-L6-v2",
        model_id    = "sentence-transformers/all-MiniLM-L6-v2",
        role        = "Task classifier — routes input to correct agent",
        size_gb     = 0.08,
        context_length = 512,
        temperature = 0.0,
        max_tokens  = 512,
        notes       = "Runs via sentence-transformers, NOT LM Studio. Fine-tune on CLINC150.",
    ),

    # --- Router / Validator (fast, lightweight) ---
    "router": ModelConfig(
        name        = "Phi-4-mini",
        model_id    = "microsoft/phi-4-mini",
        role        = "Router, validator, quick single-turn tasks",
        size_gb     = 2.5,
        context_length = 16384,
        temperature = 0.3,
        max_tokens  = 1024,
        notes       = "First model to load. Fine-tune on Dolly-15K.",
    ),

    # --- Fast Fallback ---
    "fallback": ModelConfig(
        name        = "Mistral-7B-Instruct",
        model_id    = "mistralai/mistral-7b-instruct-v0.3",
        role        = "Fast fallback for general tasks",
        size_gb     = 4.1,
        context_length = 32768,
        temperature = 0.7,
        max_tokens  = 2048,
        notes       = "Use when other agents are busy or fail.",
    ),

    # --- Code Agent (main) ---
    "coder": ModelConfig(
        name        = "Qwen2.5-Coder-7B",
        model_id    = "qwen/qwen2.5-coder-7b-instruct",
        role        = "Main code agent — write, fix, review code",
        size_gb     = 4.5,
        context_length = 32768,
        temperature = 0.2,
        max_tokens  = 4096,
        notes       = "Fine-tune on Magicoder-OSS-Instruct-75K.",
    ),

    # --- Orchestrator / Critic ---
    "orchestrator": ModelConfig(
        name        = "DeepSeek-R1-7B",
        model_id    = "deepseek-ai/deepseek-r1-distill-qwen-7b",
        role        = "Orchestrator, critic, complex reasoning",
        size_gb     = 4.7,
        context_length = 32768,
        temperature = 0.6,
        max_tokens  = 4096,
        notes       = "Used in ensemble debate engine.",
    ),

    # --- Docs / Summaries ---
    "docs": ModelConfig(
        name        = "Llama-3.3-8B",
        model_id    = "meta-llama/llama-3.3-8b-instruct",
        role        = "Documentation writer, summariser",
        size_gb     = 5.0,
        context_length = 131072,
        temperature = 0.7,
        max_tokens  = 2048,
        notes       = "Best for long-context summarisation.",
    ),

    # --- Planner / Enhancer ---
    "planner": ModelConfig(
        name        = "Qwen3-8B",
        model_id    = "qwen/qwen3-8b",
        role        = "Planner, task decomposer, enhancer",
        size_gb     = 5.2,
        context_length = 32768,
        temperature = 0.5,
        max_tokens  = 4096,
        notes       = "Fine-tune on UltraChat-200K subset.",
    ),

    # --- Voice Input ---
    "stt": ModelConfig(
        name        = "Faster-Whisper-base",
        model_id    = "guillaumekln/faster-whisper-base",
        role        = "Speech to text — voice input",
        size_gb     = 0.15,
        context_length = 0,
        temperature = 0.0,
        max_tokens  = 0,
        notes       = "Runs via faster-whisper library, NOT LM Studio.",
    ),

    # --- Voice Output ---
    "tts": ModelConfig(
        name        = "Kokoro",
        model_id    = "hexgrad/kokoro-82m",
        role        = "Text to speech — Kriya voice output",
        size_gb     = 0.082,
        context_length = 0,
        temperature = 0.0,
        max_tokens  = 0,
        notes       = "Fine-tune for custom Kriya voice persona.",
    ),
}


# ============================================================
#  RAM BUDGET (your machine: ~8GB usable at any time)
#  Never load more than 2 LLMs simultaneously.
# ============================================================

RAM_BUDGET_GB: float = 8.0
MAX_SIMULTANEOUS_MODELS: int = 2


# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def get_model(role: str) -> ModelConfig:
    """
    Fetch a model config by its role key.

    Usage:
        cfg = get_model("coder")
        print(cfg.model_id)
    """
    if role not in MODELS:
        available = list(MODELS.keys())
        raise KeyError(f"Model role '{role}' not found. Available: {available}")
    return MODELS[role]


def list_models() -> list[dict]:
    """
    Return a summary list of all models — useful for the /models API endpoint
    and the Models Panel in the GUI.
    """
    return [
        {
            "role":         role,
            "name":         cfg.name,
            "model_id":     cfg.model_id,
            "size_gb":      cfg.size_gb,
            "is_finetuned": cfg.is_finetuned,
            "adapter_path": cfg.adapter_path,
            "notes":        cfg.notes,
        }
        for role, cfg in MODELS.items()
    ]


def get_lm_studio_models() -> list[ModelConfig]:
    """
    Return only models that run through LM Studio
    (excludes classifier, stt, tts which use their own libraries).
    """
    excluded = {"classifier", "stt", "tts"}
    return [cfg for role, cfg in MODELS.items() if role not in excluded]


def mark_finetuned(role: str, adapter_path: str) -> None:
    """
    Call this after downloading a fine-tuned adapter from HuggingFace.
    Updates the registry so ai_core.py knows to load the adapter.

    Usage:
        mark_finetuned("coder", "finetuning/adapters/qwen-coder-kriya")
    """
    if role not in MODELS:
        raise KeyError(f"Model role '{role}' not found.")
    MODELS[role].is_finetuned = True
    MODELS[role].adapter_path = adapter_path
    print(f"[registry] '{role}' marked as fine-tuned. Adapter: {adapter_path}")


def total_size_gb(roles: list[str]) -> float:
    """
    Calculate total RAM needed to load a list of models.
    Use before loading to check against RAM_BUDGET_GB.

    Usage:
        size = total_size_gb(["router", "coder"])
        if size <= RAM_BUDGET_GB:
            ...
    """
    return sum(MODELS[r].size_gb for r in roles if r in MODELS)


# ============================================================
#  QUICK TEST — run this file directly to verify
#  python core/model_registry.py
# ============================================================

if __name__ == "__main__":
    from rich.table import Table
    from rich.console import Console

    console = Console()

    table = Table(title="KriyaOS Model Registry", show_lines=True)
    table.add_column("Role",       style="cyan",  no_wrap=True)
    table.add_column("Name",       style="white")
    table.add_column("Size (GB)",  style="yellow", justify="right")
    table.add_column("Context",    style="green",  justify="right")
    table.add_column("Fine-tuned", style="magenta")
    table.add_column("Role desc",  style="dim")

    for role, cfg in MODELS.items():
        table.add_row(
            role,
            cfg.name,
            str(cfg.size_gb),
            f"{cfg.context_length:,}" if cfg.context_length > 0 else "—",
            "yes" if cfg.is_finetuned else "no",
            cfg.role,
        )

    console.print(table)
    console.print(f"\n[bold]Total models:[/bold] {len(MODELS)}")
    console.print(f"[bold]RAM budget:[/bold] {RAM_BUDGET_GB} GB")
    console.print(f"[bold]Max simultaneous:[/bold] {MAX_SIMULTANEOUS_MODELS}")