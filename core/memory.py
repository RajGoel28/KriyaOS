# ============================================================
#  KriyaOS — core/memory.py
#  Persistent memory using SQLite.
#  Saves every conversation turn, task, and agent output to disk.
#  KriyaChat and all agents import from here for long-term memory.
# ============================================================

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================
#  DATABASE LOCATION
#  Stored in KriyaOS/data/memory.db — never committed to Git
# ============================================================

DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


# ============================================================
#  SETUP — creates tables if they don't exist
# ============================================================

def init_db() -> None:
    """
    Initialize the SQLite database and create all tables.
    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    Called automatically when this module is imported.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""

            -- Every conversation message (user + assistant turns)
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,   -- 'user' or 'assistant'
                content     TEXT    NOT NULL,
                model_role  TEXT,               -- which KriyaOS model replied
                created_at  TEXT    NOT NULL
            );

            -- Every task sent to KriyaOS (ask, code, build, fix, etc.)
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type   TEXT    NOT NULL,   -- 'ask', 'code', 'fix', 'build', etc.
                prompt      TEXT    NOT NULL,
                response    TEXT,
                model_role  TEXT,
                duration_ms INTEGER,            -- how long it took
                status      TEXT    DEFAULT 'done',  -- 'done', 'failed', 'pending'
                created_at  TEXT    NOT NULL
            );

            -- Agent outputs saved as fine-tuning training data
            CREATE TABLE IF NOT EXISTS training_data (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type   TEXT    NOT NULL,
                prompt      TEXT    NOT NULL,
                response    TEXT    NOT NULL,
                model_role  TEXT,
                quality     INTEGER DEFAULT 0,  -- 0=unrated, 1=bad, 2=ok, 3=good
                created_at  TEXT    NOT NULL
            );

            -- Key-value store for settings and state
            CREATE TABLE IF NOT EXISTS kv_store (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

        """)
        conn.commit()


# Auto-initialize when module is imported
init_db()


# ============================================================
#  HELPERS
# ============================================================

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows as dicts
    return conn


# ============================================================
#  MESSAGES — conversation history
# ============================================================

def save_message(
    session_id: str,
    role:       str,
    content:    str,
    model_role: Optional[str] = None,
) -> int:
    """
    Save a single conversation message to the database.

    Args:
        session_id: Unique ID for this conversation session
        role:       'user' or 'assistant'
        content:    The message text
        model_role: Which KriyaOS model replied (e.g. 'coder', 'router')

    Returns:
        The row ID of the saved message

    Usage:
        save_message("session_001", "user", "Write a FastAPI server")
        save_message("session_001", "assistant", "Here is...", model_role="coder")
    """
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, model_role, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, model_role, _now())
        )
        conn.commit()
        return cursor.lastrowid


def load_history(session_id: str, limit: int = 50) -> list[dict]:
    """
    Load conversation history for a session.
    Returns messages in chronological order, ready to pass to ai_core.

    Args:
        session_id: The session to load
        limit:      Max number of messages to return (most recent)

    Returns:
        List of {"role": ..., "content": ...} dicts

    Usage:
        history = load_history("session_001")
        chat = KriyaChat(role="coder")
        chat.history = history
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()

    # Reverse to get chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def list_sessions() -> list[dict]:
    """
    List all conversation sessions with message count and last activity.

    Returns:
        List of session summaries
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                session_id,
                COUNT(*) as message_count,
                MAX(created_at) as last_active
            FROM messages
            GROUP BY session_id
            ORDER BY last_active DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    """Delete all messages for a session."""
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
    console.print(f"[dim][ session '{session_id}' deleted ][/dim]")


# ============================================================
#  TASKS — log every KriyaOS task for fine-tuning data
# ============================================================

def save_task(
    task_type:   str,
    prompt:      str,
    response:    str,
    model_role:  Optional[str] = None,
    duration_ms: Optional[int] = None,
    status:      str = "done",
) -> int:
    """
    Save a completed task. Every ask/code/fix/build call logs here.
    This is your automatic fine-tuning dataset collector.

    Args:
        task_type:   'ask', 'code', 'fix', 'build', 'design', etc.
        prompt:      The input prompt
        response:    The model's response
        model_role:  Which model handled it
        duration_ms: How long it took in milliseconds
        status:      'done', 'failed', or 'pending'

    Returns:
        The row ID of the saved task

    Usage:
        import time
        start = time.time()
        response = ask("Write a sort function", role="coder")
        ms = int((time.time() - start) * 1000)
        save_task("code", "Write a sort function", response, "coder", ms)
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO tasks (task_type, prompt, response, model_role, duration_ms, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (task_type, prompt, response, model_role, duration_ms, status, _now())
        )
        conn.commit()

        # Also save to training_data automatically
        conn.execute(
            """
            INSERT INTO training_data (task_type, prompt, response, model_role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_type, prompt, response, model_role, _now())
        )
        conn.commit()
        return cursor.lastrowid


def load_tasks(task_type: Optional[str] = None, limit: int = 100) -> list[dict]:
    """
    Load saved tasks, optionally filtered by type.

    Usage:
        all_tasks  = load_tasks()
        code_tasks = load_tasks("code")
    """
    with _connect() as conn:
        if task_type:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE task_type = ? ORDER BY id DESC LIMIT ?",
                (task_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ============================================================
#  TRAINING DATA — export for Kaggle fine-tuning
# ============================================================

def export_training_data(
    task_type:  Optional[str] = None,
    output_path: Optional[Path] = None,
    min_quality: int = 0,
) -> Path:
    """
    Export training data to a JSONL file for Kaggle fine-tuning.
    Each line is one training example in the format Unsloth expects.

    Args:
        task_type:   Filter by task type (None = all)
        output_path: Where to save the file (default: finetuning/dataset/)
        min_quality: Only export rows with quality >= this (0 = all)

    Returns:
        Path to the exported file

    Usage:
        path = export_training_data("code")
        print(f"Exported to {path}")
        # Upload this file to Kaggle for fine-tuning
    """
    if output_path is None:
        export_dir = Path(__file__).parent.parent / "finetuning" / "dataset"
        export_dir.mkdir(parents=True, exist_ok=True)
        label       = task_type or "all"
        output_path = export_dir / f"{label}_tasks.jsonl"

    with _connect() as conn:
        if task_type:
            rows = conn.execute(
                "SELECT * FROM training_data WHERE task_type = ? AND quality >= ?",
                (task_type, min_quality)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM training_data WHERE quality >= ?",
                (min_quality,)
            ).fetchall()

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            # Unsloth / Alpaca format
            example = {
                "instruction": row["prompt"],
                "output":      row["response"],
                "task_type":   row["task_type"],
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    console.print(f"[green]Exported {len(rows)} examples → {output_path}[/green]")
    return output_path


def rate_training_example(row_id: int, quality: int) -> None:
    """
    Rate a training example for quality filtering before fine-tuning.
    1 = bad, 2 = ok, 3 = good

    Usage:
        rate_training_example(42, quality=3)
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE training_data SET quality = ? WHERE id = ?",
            (quality, row_id)
        )
        conn.commit()


# ============================================================
#  KEY-VALUE STORE — settings and persistent state
# ============================================================

def kv_set(key: str, value) -> None:
    """
    Save any value to the key-value store.

    Usage:
        kv_set("default_model", "coder")
        kv_set("theme", "dark")
        kv_set("last_session", "session_001")
    """
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), _now())
        )
        conn.commit()


def kv_get(key: str, default=None):
    """
    Get a value from the key-value store.

    Usage:
        model = kv_get("default_model", default="router")
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def kv_delete(key: str) -> None:
    """Delete a key from the store."""
    with _connect() as conn:
        conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        conn.commit()


# ============================================================
#  STATS — quick database summary
# ============================================================

def stats() -> dict:
    """
    Return a summary of everything stored in the database.

    Usage:
        from core.memory import stats
        print(stats())
    """
    with _connect() as conn:
        return {
            "total_messages":      conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "total_sessions":      conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0],
            "total_tasks":         conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
            "total_training_data": conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0],
            "db_path":             str(DB_PATH),
        }


# ============================================================
#  QUICK TEST — run this file directly to verify
#  python core/memory.py
# ============================================================

if __name__ == "__main__":
    console.rule("[bold cyan]KriyaOS — memory.py test[/bold cyan]")

    # 1. Save some messages
    console.print("\n[1] Saving messages...")
    save_message("test_session", "user",      "Write a Python sort function")
    save_message("test_session", "assistant", "Here is a sort function: ...", model_role="coder")
    save_message("test_session", "user",      "Now add type hints")
    save_message("test_session", "assistant", "Here it is with type hints: ...", model_role="coder")
    console.print("[green]Messages saved.[/green]")

    # 2. Load history
    console.print("\n[2] Loading history...")
    history = load_history("test_session")
    for msg in history:
        console.print(f"  [{msg['role']}]: {msg['content'][:60]}")

    # 3. Save a task
    console.print("\n[3] Saving task...")
    save_task("code", "Write a sort function", "def sort(lst): return sorted(lst)", "coder", 1200)
    console.print("[green]Task saved.[/green]")

    # 4. KV store
    console.print("\n[4] Testing KV store...")
    kv_set("default_model", "coder")
    kv_set("last_session",  "test_session")
    console.print(f"  default_model = {kv_get('default_model')}")
    console.print(f"  last_session  = {kv_get('last_session')}")

    # 5. Export training data
    console.print("\n[5] Exporting training data...")
    path = export_training_data("code")
    console.print(f"  Saved to: {path}")

    # 6. Stats
    console.print("\n[6] Database stats:")
    s = stats()
    table = Table(show_header=False)
    table.add_column("Key",   style="cyan")
    table.add_column("Value", style="green")
    for k, v in s.items():
        table.add_row(str(k), str(v))
    console.print(table)

    # 7. Cleanup test session
    delete_session("test_session")

    console.rule("[bold green]All memory tests passed![/bold green]")