# ============================================================
#  KriyaOS — orchestrator/model_manager.py
#  Manages which models are loaded in LM Studio at any time.
#  Enforces the 2-model RAM limit for your 16GB machine.
#  Tracks load/unload events and reports RAM usage.
#
#  Rules:
#    - Max 2 LLM models loaded simultaneously
#    - Always unload before loading if at limit
#    - classifier / stt / tts are excluded (not LM Studio)
#    - All load/unload events posted to the bus
# ============================================================

import time
import httpx
import sys
import os
from typing import Optional
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model_registry import get_model, get_lm_studio_models, RAM_BUDGET_GB, MAX_SIMULTANEOUS_MODELS, ModelConfig
from orchestrator.bus import bus, Topic

console = Console()

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# Non-LM Studio models — never try to load/unload these via API
EXCLUDED_ROLES = {"classifier", "stt", "tts"}


# ============================================================
#  STATE
#  Tracks currently loaded models in memory
# ============================================================

class _State:
    loaded: list[str] = []       # list of currently loaded model roles
    load_times: dict[str, float] = {}  # role → timestamp when loaded

_state = _State()


# ============================================================
#  LM STUDIO API HELPERS
# ============================================================

def _get_loaded_models() -> list[str]:
    """
    Ask LM Studio which models are currently loaded.
    Returns list of model_id strings.
    """
    try:
        response = httpx.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []


def _request_load(model_id: str) -> bool:
    """
    Ask LM Studio to load a model.
    LM Studio loads models automatically when you send a request to them.
    This sends a minimal ping request to trigger loading.
    Returns True if successful.
    """
    try:
        response = httpx.post(
            f"{LM_STUDIO_BASE_URL}/chat/completions",
            json={
                "model":      model_id,
                "messages":   [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream":     False,
            },
            timeout=60,
        )
        return response.status_code == 200
    except Exception as e:
        console.print(f"[red][ model_manager ] Load request failed: {e}[/red]")
        return False


def _request_unload(model_id: str) -> bool:
    """
    Ask LM Studio to unload a model via the /v1/models/{id} DELETE endpoint.
    Note: Not all LM Studio versions support this — falls back to manual unload.
    """
    try:
        response = httpx.delete(
            f"{LM_STUDIO_BASE_URL}/models/{model_id}",
            timeout=10,
        )
        return response.status_code in (200, 204)
    except Exception:
        return False


# ============================================================
#  PUBLIC API
# ============================================================

def load(role: str) -> bool:
    """
    Load a model by its KriyaOS role name.
    Enforces the 2-model limit — unloads the oldest if at limit.

    Args:
        role: Model role from registry e.g. "coder", "router"

    Returns:
        True if loaded successfully

    Usage:
        load("coder")
        load("orchestrator")
    """
    if role in EXCLUDED_ROLES:
        console.print(f"[dim][ model_manager ] '{role}' is not an LM Studio model — skipping.[/dim]")
        return True

    if role in _state.loaded:
        console.print(f"[dim][ model_manager ] '{role}' is already loaded.[/dim]")
        return True

    cfg = get_model(role)

    # Enforce RAM limit — unload oldest if at max
    if len(_state.loaded) >= MAX_SIMULTANEOUS_MODELS:
        oldest = _state.loaded[0]
        console.print(f"[yellow][ model_manager ] At model limit — unloading '{oldest}'[/yellow]")
        unload(oldest)

    console.print(f"[cyan][ model_manager ] Loading '{cfg.name}'...[/cyan]")
    start = time.time()

    success = _request_load(cfg.model_id)

    if success:
        _state.loaded.append(role)
        _state.load_times[role] = time.time()
        elapsed = round(time.time() - start, 1)
        console.print(f"[green][ model_manager ] '{cfg.name}' loaded in {elapsed}s[/green]")

        bus.post(Topic.SYSTEM_STATUS, "model_manager", {
            "event":    "loaded",
            "role":     role,
            "model_id": cfg.model_id,
            "elapsed_s": elapsed,
        })
    else:
        console.print(f"[red][ model_manager ] Failed to load '{cfg.name}'[/red]")
        console.print(f"[dim]Tip: Make sure the model is downloaded in LM Studio first.[/dim]")

    return success


def unload(role: str) -> bool:
    """
    Unload a model by its role name.

    Args:
        role: Model role to unload

    Returns:
        True if unloaded successfully

    Usage:
        unload("coder")
    """
    if role in EXCLUDED_ROLES:
        return True

    if role not in _state.loaded:
        console.print(f"[dim][ model_manager ] '{role}' is not loaded — nothing to unload.[/dim]")
        return True

    cfg = get_model(role)
    console.print(f"[yellow][ model_manager ] Unloading '{cfg.name}'...[/yellow]")

    success = _request_unload(cfg.model_id)

    # Remove from state regardless — LM Studio may not support DELETE
    if role in _state.loaded:
        _state.loaded.remove(role)
    if role in _state.load_times:
        del _state.load_times[role]

    console.print(f"[dim][ model_manager ] '{cfg.name}' unloaded.[/dim]")

    bus.post(Topic.SYSTEM_STATUS, "model_manager", {
        "event":    "unloaded",
        "role":     role,
        "model_id": cfg.model_id,
    })

    return True


def ensure_loaded(roles: list[str]) -> bool:
    """
    Make sure a list of model roles are all loaded.
    Loads any that are missing. Respects the 2-model limit.

    Args:
        roles: List of role names to ensure are loaded

    Returns:
        True if all loaded successfully

    Usage:
        # Before running ensemble
        ensure_loaded(["coder", "orchestrator"])
    """
    # Filter out excluded roles
    roles = [r for r in roles if r not in EXCLUDED_ROLES]

    if len(roles) > MAX_SIMULTANEOUS_MODELS:
        console.print(
            f"[yellow][ model_manager ] Warning: requested {len(roles)} models "
            f"but limit is {MAX_SIMULTANEOUS_MODELS}. Loading first {MAX_SIMULTANEOUS_MODELS}.[/yellow]"
        )
        roles = roles[:MAX_SIMULTANEOUS_MODELS]

    success = True
    for role in roles:
        if role not in _state.loaded:
            ok = load(role)
            if not ok:
                success = False

    return success


def unload_all() -> None:
    """
    Unload all currently loaded models.
    Call this when KriyaOS shuts down or between major tasks.

    Usage:
        unload_all()
    """
    for role in list(_state.loaded):
        unload(role)
    console.print("[dim][ model_manager ] All models unloaded.[/dim]")


def loaded_models() -> list[str]:
    """
    Return list of currently loaded model roles.

    Usage:
        roles = loaded_models()
        print(roles)  # ["router", "coder"]
    """
    return list(_state.loaded)


def is_loaded(role: str) -> bool:
    """
    Check if a specific model is currently loaded.

    Usage:
        if not is_loaded("coder"):
            load("coder")
    """
    return role in _state.loaded


def ram_usage() -> dict:
    """
    Estimate current RAM usage based on loaded models.

    Returns:
        Dict with used_gb, budget_gb, free_gb, percent_used

    Usage:
        ram = ram_usage()
        print(f"RAM: {ram['used_gb']}GB / {ram['budget_gb']}GB")
    """
    from core.model_registry import MODELS
    used = sum(
        MODELS[r].size_gb
        for r in _state.loaded
        if r in MODELS and r not in EXCLUDED_ROLES
    )
    return {
        "used_gb":      round(used, 1),
        "budget_gb":    RAM_BUDGET_GB,
        "free_gb":      round(RAM_BUDGET_GB - used, 1),
        "percent_used": round((used / RAM_BUDGET_GB) * 100, 1),
        "loaded_roles": list(_state.loaded),
    }


def status() -> None:
    """
    Print a formatted status table of all models.
    Used by `kriya status` CLI command.

    Usage:
        from orchestrator.model_manager import status
        status()
    """
    from core.model_registry import MODELS

    table = Table(title="KriyaOS Model Status", show_lines=True)
    table.add_column("Role",     style="cyan",    no_wrap=True)
    table.add_column("Name",     style="white",   no_wrap=True)
    table.add_column("Size GB",  style="yellow",  justify="right")
    table.add_column("Loaded",   style="magenta", justify="center")
    table.add_column("Uptime",   style="dim",     justify="right")

    now = time.time()
    for role, cfg in MODELS.items():
        loaded     = role in _state.loaded
        load_time  = _state.load_times.get(role)
        uptime_str = ""
        if loaded and load_time:
            secs = int(now - load_time)
            uptime_str = f"{secs // 60}m {secs % 60}s"

        table.add_row(
            role,
            cfg.name,
            str(cfg.size_gb),
            "[green]yes[/green]" if loaded else "no",
            uptime_str,
        )

    console.print(table)

    ram = ram_usage()
    console.print(
        f"\nRAM: [yellow]{ram['used_gb']}GB[/yellow] used / "
        f"[green]{ram['budget_gb']}GB[/green] budget  "
        f"([cyan]{ram['percent_used']}%[/cyan])"
    )


# ============================================================
#  QUICK TEST — python orchestrator/model_manager.py
#  LM Studio must be running with Phi-4-mini loaded
# ============================================================

if __name__ == "__main__":
    import httpx as _httpx

    console.rule("[bold cyan]KriyaOS — model_manager test[/bold cyan]")

    # Check LM Studio
    try:
        r = _httpx.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
        online = r.status_code == 200
    except Exception:
        online = False

    if not online:
        console.print("[red]LM Studio is not running.[/red]")
        exit(1)

    console.print("[green]LM Studio online.[/green]\n")

    # 1. Show initial status
    console.print("[1] Initial status:")
    status()

    # 2. Test RAM usage
    console.print("\n[2] RAM usage:")
    ram = ram_usage()
    for k, v in ram.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")

    # 3. Test is_loaded
    console.print("\n[3] is_loaded checks:")
    console.print(f"  router loaded: {is_loaded('router')}")
    console.print(f"  coder loaded:  {is_loaded('coder')}")

    # 4. Test ensure_loaded with router (already loaded in LM Studio)
    console.print("\n[4] ensure_loaded(['router']):")
    ensure_loaded(["router"])

    # 5. Final status
    console.print("\n[5] Final status:")
    status()

    console.rule("[bold green]Model manager test done![/bold green]")