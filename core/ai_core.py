# ============================================================
#  KriyaOS — core/ai_core.py
#  The brain. Talks to LM Studio API, handles all LLM calls.
#  Supports: single ask, streaming, multi-turn chat.
#  Every agent and CLI command goes through this file.
# ============================================================

import json
import httpx
from typing import Generator, Optional
from rich.console import Console
from rich.live import Live
from rich.text import Text

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model_registry import ModelConfig, get_model, MODELS

# ============================================================
#  CONFIG
# ============================================================

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT     = 120   # seconds — long tasks need more time
STREAM_TIMEOUT      = 300   # seconds — streaming can take longer

console = Console()


# ============================================================
#  CONNECTION CHECK
# ============================================================

def is_online() -> bool:
    """
    Check if LM Studio server is running on localhost:1234.
    Always call this before sending any request.

    Usage:
        if not is_online():
            print("Start LM Studio first!")
    """
    try:
        response = httpx.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def require_online() -> None:
    """
    Raise a clear error if LM Studio is not running.
    Called at the start of ask(), stream(), chat().
    """
    if not is_online():
        raise ConnectionError(
            "\n[KriyaOS] LM Studio is not running!\n"
            "  1. Open LM Studio\n"
            "  2. Load a model\n"
            "  3. Go to Local Server tab → Start Server\n"
            "  4. Try again\n"
        )


# ============================================================
#  BUILD MESSAGE PAYLOAD
# ============================================================

def _build_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Build the messages array for the LM Studio API.

    Args:
        prompt:        The user's message
        system_prompt: Optional system instruction (sets model behaviour)
        history:       Optional list of previous messages for multi-turn chat
                       Format: [{"role": "user", "content": "..."}, ...]

    Returns:
        List of message dicts ready for the API
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    return messages


def _build_payload(
    messages:    list[dict],
    cfg:         ModelConfig,
    stream:      bool = False,
    max_tokens:  Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    """
    Build the full request payload for LM Studio API.
    Uses model config defaults, overrides if provided.
    """
    return {
        "model":       cfg.model_id,
        "messages":    messages,
        "temperature": temperature if temperature is not None else cfg.temperature,
        "max_tokens":  max_tokens  if max_tokens  is not None else cfg.max_tokens,
        "stream":      stream,
    }


# ============================================================
#  ASK — single prompt, full response (no streaming)
# ============================================================

def ask(
    prompt:        str,
    role:          str = "router",
    system_prompt: Optional[str] = None,
    max_tokens:    Optional[int] = None,
    temperature:   Optional[float] = None,
) -> str:
    """
    Send a single prompt to a model. Wait for the full response.
    Use this for quick tasks, routing decisions, validations.

    Args:
        prompt:        The question or instruction
        role:          Model role from registry (default: "router" = Phi-4-mini)
        system_prompt: Optional system instruction
        max_tokens:    Override default max tokens
        temperature:   Override default temperature

    Returns:
        The model's response as a string

    Usage:
        response = ask("What is Python?")
        response = ask("Write a function to sort a list", role="coder")
    """
    require_online()

    cfg      = get_model(role)
    messages = _build_messages(prompt, system_prompt)
    payload  = _build_payload(messages, cfg, stream=False, max_tokens=max_tokens, temperature=temperature)

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            response = client.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    except httpx.TimeoutException:
        raise TimeoutError(f"[KriyaOS] Model '{cfg.name}' timed out after {DEFAULT_TIMEOUT}s.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"[KriyaOS] LM Studio API error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"[KriyaOS] Unexpected error in ask(): {e}")


# ============================================================
#  STREAM — single prompt, word-by-word streaming
# ============================================================

def stream(
    prompt:        str,
    role:          str = "router",
    system_prompt: Optional[str] = None,
    max_tokens:    Optional[int] = None,
    temperature:   Optional[float] = None,
    print_output:  bool = True,
) -> Generator[str, None, str]:
    """
    Send a prompt and stream the response token by token.
    Yields each chunk as it arrives. Optionally prints live to terminal.

    Args:
        prompt:        The question or instruction
        role:          Model role from registry
        system_prompt: Optional system instruction
        max_tokens:    Override default max tokens
        temperature:   Override default temperature
        print_output:  If True, prints streaming output live in terminal

    Yields:
        Each text chunk (token) as it arrives

    Returns:
        Full assembled response string when done

    Usage:
        for chunk in stream("Explain Python decorators", role="coder"):
            pass  # chunks are printed automatically

        # Or collect the full response:
        full = ""
        for chunk in stream("Write a FastAPI server", role="coder", print_output=False):
            full += chunk
    """
    require_online()

    cfg      = get_model(role)
    messages = _build_messages(prompt, system_prompt)
    payload  = _build_payload(messages, cfg, stream=True, max_tokens=max_tokens, temperature=temperature)

    full_response = ""

    try:
        with httpx.Client(timeout=STREAM_TIMEOUT) as client:
            with client.stream(
                "POST",
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()

                if print_output:
                    console.print(f"\n[dim][{role} → {cfg.name}][/dim]", end=" ")

                for line in response.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        line = line[6:]  # strip "data: " prefix

                    try:
                        chunk_data = json.loads(line)
                        delta      = chunk_data["choices"][0].get("delta", {})
                        content    = delta.get("content", "")

                        if content:
                            full_response += content
                            if print_output:
                                print(content, end="", flush=True)
                            yield content

                    except (json.JSONDecodeError, KeyError):
                        continue

                if print_output:
                    print()  # newline after streaming ends

    except httpx.TimeoutException:
        raise TimeoutError(f"[KriyaOS] Stream timed out after {STREAM_TIMEOUT}s.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"[KriyaOS] LM Studio API error: {e.response.status_code}")
    except Exception as e:
        raise RuntimeError(f"[KriyaOS] Unexpected error in stream(): {e}")

    return full_response


# ============================================================
#  CHAT — multi-turn conversation with history
# ============================================================

class KriyaChat:
    """
    Multi-turn conversation session with memory.
    Keeps the full conversation history in memory for the session.
    For persistent memory across sessions, see core/memory.py.

    Usage:
        chat = KriyaChat(role="coder")
        chat.say("Hi, I need help with Python")
        chat.say("Write a function to reverse a string")
        chat.say("Now add type hints to it")
        chat.clear()  # reset conversation
    """

    def __init__(
        self,
        role:          str = "router",
        system_prompt: Optional[str] = None,
        streaming:     bool = True,
    ):
        self.role          = role
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.streaming     = streaming
        self.history:      list[dict] = []
        self.cfg           = get_model(role)

    def _default_system_prompt(self) -> str:
        return (
            "You are Kriya, an intelligent AI assistant built into KriyaOS. "
            "You are helpful, concise, and technical. "
            "You assist with coding, planning, debugging, and building software."
        )

    def say(
        self,
        prompt:     str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a message. Response is added to history automatically.

        Args:
            prompt:     Your message
            max_tokens: Override max tokens for this turn

        Returns:
            The model's response as a string
        """
        require_online()

        messages = _build_messages(prompt, self.system_prompt, self.history)
        payload  = _build_payload(
            messages, self.cfg,
            stream=self.streaming,
            max_tokens=max_tokens,
        )

        if self.streaming:
            response_text = ""
            console.print(f"\n[dim][{self.cfg.name}][/dim]", end=" ")

            with httpx.Client(timeout=STREAM_TIMEOUT) as client:
                with client.stream(
                    "POST",
                    f"{LM_STUDIO_BASE_URL}/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            chunk_data = json.loads(line)
                            delta      = chunk_data["choices"][0].get("delta", {})
                            content    = delta.get("content", "")
                            if content:
                                response_text += content
                                print(content, end="", flush=True)
                        except (json.JSONDecodeError, KeyError):
                            continue

            print()

        else:
            response_text = ask(
                prompt,
                role=self.role,
                system_prompt=self.system_prompt,
                max_tokens=max_tokens,
            )
            console.print(f"\n[dim][{self.cfg.name}][/dim] {response_text}")

        # Save to history
        self.history.append({"role": "user",      "content": prompt})
        self.history.append({"role": "assistant",  "content": response_text})

        return response_text

    def clear(self) -> None:
        """Reset conversation history."""
        self.history = []
        console.print("[dim][ conversation cleared ][/dim]")

    def show_history(self) -> None:
        """Print the full conversation history."""
        if not self.history:
            console.print("[dim]No history yet.[/dim]")
            return
        for msg in self.history:
            role  = msg["role"].upper()
            color = "cyan" if role == "USER" else "green"
            console.print(f"[{color}]{role}:[/{color}] {msg['content']}\n")


# ============================================================
#  QUICK TEST — run this file directly to verify
#  python core/ai_core.py
# ============================================================

if __name__ == "__main__":
    console.rule("[bold cyan]KriyaOS — ai_core test[/bold cyan]")

    # 1. Check LM Studio is running
    console.print("\n[1] Checking LM Studio connection...")
    if not is_online():
        console.print("[red]LM Studio is not running.[/red]")
        console.print("Start LM Studio → load a model → Local Server → Start Server")
        exit(1)
    console.print("[green]LM Studio is online.[/green]")

    # 2. Single ask test
    console.print("\n[2] Testing ask()...")
    response = ask("Reply with exactly 3 words: KriyaOS is ready", role="router")
    console.print(f"Response: [green]{response}[/green]")

    # 3. Streaming test
    console.print("\n[3] Testing stream()...")
    console.print("Prompt: [cyan]What is KriyaOS in one sentence?[/cyan]")
    for _ in stream("What is KriyaOS in one sentence?", role="router"):
        pass

    # 4. Multi-turn chat test
    console.print("\n[4] Testing KriyaChat (2 turns)...")
    chat = KriyaChat(role="router")
    chat.say("My name is RajBHAI. Remember it.")
    chat.say("What is my name?")

    console.rule("[bold green]All tests passed![/bold green]")