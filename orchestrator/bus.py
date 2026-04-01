# ============================================================
#  KriyaOS — orchestrator/bus.py
#  Agent message bus — shared communication layer.
#  All agents talk to each other through the bus.
#  No agent imports another agent directly — they post messages
#  to the bus and subscribe to results. Keeps agents decoupled.
#
#  Flow:
#    agent_a.post("task.code", payload)
#      → bus stores message
#      → agent_b picks it up via bus.get("task.code")
#      → agent_b posts result to bus.post("result.code", result)
#      → orchestrator reads bus.get("result.code")
# ============================================================

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()


# ============================================================
#  MESSAGE
# ============================================================

@dataclass
class Message:
    """
    A single message on the bus.
    Every agent communication goes through this structure.
    """
    id:          str              # unique message ID
    topic:       str              # what this message is about e.g. "task.code"
    sender:      str              # which agent sent it
    payload:     Any              # the actual data
    timestamp:   float            # unix timestamp
    read:        bool = False     # has this been consumed?
    reply_to:    Optional[str] = None  # ID of message this is replying to
    metadata:    dict = field(default_factory=dict)  # extra info


def new_message(
    topic:    str,
    sender:   str,
    payload:  Any,
    reply_to: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Message:
    """
    Create a new Message with auto-generated ID and timestamp.

    Usage:
        msg = new_message("task.code", "router", {"prompt": "Write a sort fn"})
    """
    return Message(
        id        = str(uuid.uuid4())[:8],
        topic     = topic,
        sender    = sender,
        payload   = payload,
        timestamp = time.time(),
        reply_to  = reply_to,
        metadata  = metadata or {},
    )


# ============================================================
#  BUS TOPICS
#  Standard topic names used across all KriyaOS agents.
#  Always use these constants — never hardcode topic strings.
# ============================================================

class Topic:
    # Task routing
    TASK_IN         = "task.in"          # new task arrives from user
    TASK_ROUTED     = "task.routed"      # router has made a decision

    # Agent tasks
    TASK_PLAN       = "task.plan"        # send to planner agent
    TASK_CODE       = "task.code"        # send to backend_agent
    TASK_FIX        = "task.fix"         # send to backend_agent
    TASK_REVIEW     = "task.review"      # send to reviewer agent
    TASK_DOCS       = "task.docs"        # send to docs_agent
    TASK_DESIGN     = "task.design"      # send to design_agent

    # Results from agents
    RESULT_PLAN     = "result.plan"
    RESULT_CODE     = "result.code"
    RESULT_FIX      = "result.fix"
    RESULT_REVIEW   = "result.review"
    RESULT_DOCS     = "result.docs"
    RESULT_DESIGN   = "result.design"

    # Ensemble pipeline
    ENSEMBLE_DRAFT  = "ensemble.draft"   # first model output
    ENSEMBLE_CRITIQUE = "ensemble.critique"  # critic's feedback
    ENSEMBLE_FINAL  = "ensemble.final"   # polished final output

    # System
    SYSTEM_ERROR    = "system.error"     # any agent error
    SYSTEM_STATUS   = "system.status"    # model load/unload events
    SYSTEM_DONE     = "system.done"      # task fully complete


# ============================================================
#  MESSAGE BUS
# ============================================================

class MessageBus:
    """
    In-memory message bus for KriyaOS agent communication.
    Thread-safe for single-process use.

    Usage:
        bus = MessageBus()

        # Post a message
        bus.post("task.code", sender="router", payload={"prompt": "..."})

        # Get latest message on a topic
        msg = bus.get("task.code")

        # Get all unread messages on a topic
        msgs = bus.get_all("task.code")

        # Subscribe callback (called whenever a message is posted)
        bus.subscribe("result.code", callback=my_handler)
    """

    def __init__(self):
        # topic → list of messages
        self._messages: dict[str, list[Message]] = defaultdict(list)
        # topic → list of subscriber callbacks
        self._subscribers: dict[str, list] = defaultdict(list)
        # full history of all messages ever posted
        self._history: list[Message] = []


    # ----------------------------------------------------------
    #  POST
    # ----------------------------------------------------------

    def post(
        self,
        topic:    str,
        sender:   str,
        payload:  Any,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Message:
        """
        Post a message to the bus on a given topic.
        Triggers any subscribers registered for that topic.

        Args:
            topic:    The topic string (use Topic constants)
            sender:   Name of the sending agent
            payload:  Any data to send
            reply_to: Optional message ID this is replying to
            metadata: Optional extra key-value data

        Returns:
            The created Message object

        Usage:
            msg = bus.post(Topic.TASK_CODE, "router", {"prompt": "Write a sort fn"})
        """
        msg = new_message(topic, sender, payload, reply_to, metadata)
        self._messages[topic].append(msg)
        self._history.append(msg)

        # Notify subscribers
        for callback in self._subscribers.get(topic, []):
            try:
                callback(msg)
            except Exception as e:
                console.print(f"[red][ bus ] Subscriber error on '{topic}': {e}[/red]")

        return msg


    # ----------------------------------------------------------
    #  GET
    # ----------------------------------------------------------

    def get(self, topic: str, mark_read: bool = True) -> Optional[Message]:
        """
        Get the latest unread message on a topic.
        Returns None if no unread messages exist.

        Args:
            topic:     The topic to check
            mark_read: Mark message as read after fetching

        Usage:
            msg = bus.get(Topic.RESULT_CODE)
            if msg:
                print(msg.payload)
        """
        messages = self._messages.get(topic, [])
        for msg in reversed(messages):
            if not msg.read:
                if mark_read:
                    msg.read = True
                return msg
        return None


    def get_all(self, topic: str, mark_read: bool = True) -> list[Message]:
        """
        Get all unread messages on a topic (oldest first).

        Usage:
            msgs = bus.get_all(Topic.TASK_REVIEW)
            for msg in msgs:
                process(msg.payload)
        """
        messages = self._messages.get(topic, [])
        unread = [m for m in messages if not m.read]
        if mark_read:
            for m in unread:
                m.read = True
        return unread


    def peek(self, topic: str) -> Optional[Message]:
        """
        Get the latest message without marking it as read.

        Usage:
            msg = bus.peek(Topic.ENSEMBLE_DRAFT)
        """
        return self.get(topic, mark_read=False)


    def wait_for(
        self,
        topic:      str,
        timeout_s:  float = 60.0,
        poll_ms:    float = 0.05,
    ) -> Optional[Message]:
        """
        Block until a message arrives on a topic or timeout is reached.
        Use this when one agent needs to wait for another to finish.

        Args:
            topic:     Topic to wait on
            timeout_s: Max seconds to wait (default 60)
            poll_ms:   How often to check in seconds (default 50ms)

        Returns:
            The message, or None if timed out

        Usage:
            # Wait up to 5 minutes for coder to finish
            result = bus.wait_for(Topic.RESULT_CODE, timeout_s=300)
            if result is None:
                print("Coder timed out!")
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            msg = self.get(topic)
            if msg:
                return msg
            time.sleep(poll_ms)
        console.print(f"[yellow][ bus ] Timeout waiting for '{topic}' after {timeout_s}s[/yellow]")
        return None


    # ----------------------------------------------------------
    #  SUBSCRIBE
    # ----------------------------------------------------------

    def subscribe(self, topic: str, callback) -> None:
        """
        Register a callback to be called whenever a message
        is posted on a topic.

        Args:
            topic:    Topic to subscribe to
            callback: Function to call with the Message object

        Usage:
            def on_code_result(msg):
                print("Code result:", msg.payload)

            bus.subscribe(Topic.RESULT_CODE, on_code_result)
        """
        self._subscribers[topic].append(callback)


    def unsubscribe(self, topic: str, callback) -> None:
        """Remove a subscriber callback."""
        if topic in self._subscribers:
            self._subscribers[topic] = [
                cb for cb in self._subscribers[topic] if cb != callback
            ]


    # ----------------------------------------------------------
    #  CLEAR + INSPECT
    # ----------------------------------------------------------

    def clear(self, topic: Optional[str] = None) -> None:
        """
        Clear messages. If topic given, clears only that topic.
        If no topic given, clears everything.

        Usage:
            bus.clear(Topic.TASK_CODE)  # clear one topic
            bus.clear()                 # clear all
        """
        if topic:
            self._messages[topic] = []
        else:
            self._messages.clear()
            self._history.clear()


    def topics(self) -> list[str]:
        """Return list of all active topics."""
        return list(self._messages.keys())


    def pending_count(self, topic: str) -> int:
        """Count unread messages on a topic."""
        return sum(1 for m in self._messages.get(topic, []) if not m.read)


    def history(self, limit: int = 50) -> list[Message]:
        """Return recent message history across all topics."""
        return self._history[-limit:]


    def stats(self) -> dict:
        """Return bus statistics."""
        total   = len(self._history)
        unread  = sum(1 for m in self._history if not m.read)
        topics  = len(self._messages)
        return {
            "total_messages":  total,
            "unread_messages": unread,
            "active_topics":   topics,
            "subscribers":     sum(len(v) for v in self._subscribers.values()),
        }


    def print_history(self, limit: int = 20) -> None:
        """Print recent message history as a table."""
        table = Table(title=f"Bus History (last {limit})", show_lines=True)
        table.add_column("ID",      style="dim",     no_wrap=True)
        table.add_column("Topic",   style="cyan",    no_wrap=True)
        table.add_column("Sender",  style="magenta", no_wrap=True)
        table.add_column("Read",    style="yellow",  justify="center")
        table.add_column("Payload", style="white",   max_width=40)

        for msg in self.history(limit):
            payload_str = str(msg.payload)[:40]
            table.add_row(
                msg.id,
                msg.topic,
                msg.sender,
                "yes" if msg.read else "no",
                payload_str,
            )
        console.print(table)


# ============================================================
#  GLOBAL BUS INSTANCE
#  Import this singleton in all agents — never create a new bus.
#
#  Usage in any agent:
#    from orchestrator.bus import bus, Topic
#    bus.post(Topic.RESULT_CODE, "backend_agent", {"code": "..."})
# ============================================================

bus = MessageBus()


# ============================================================
#  QUICK TEST — python orchestrator/bus.py
# ============================================================

if __name__ == "__main__":
    console.rule("[bold cyan]KriyaOS — bus test[/bold cyan]")

    # 1. Basic post and get
    console.print("\n[1] Basic post and get...")
    bus.post(Topic.TASK_CODE, "router", {"prompt": "Write a sort function"})
    msg = bus.get(Topic.TASK_CODE)
    assert msg is not None
    assert msg.payload["prompt"] == "Write a sort function"
    assert msg.read == True
    console.print(f"  [green]Got message: {msg.payload}[/green]")

    # 2. Unread check — same message should not appear again
    console.print("\n[2] Unread check...")
    msg2 = bus.get(Topic.TASK_CODE)
    assert msg2 is None, "Already read message should not return again"
    console.print("  [green]Read messages correctly filtered out.[/green]")

    # 3. Subscribe callback
    console.print("\n[3] Subscriber test...")
    received = []
    def on_result(msg):
        received.append(msg.payload)

    bus.subscribe(Topic.RESULT_CODE, on_result)
    bus.post(Topic.RESULT_CODE, "backend_agent", {"code": "def sort(x): return sorted(x)"})
    assert len(received) == 1
    console.print(f"  [green]Subscriber received: {received[0]}[/green]")

    # 4. Full agent conversation simulation
    console.print("\n[4] Simulating agent pipeline...")

    # Router posts task
    t = bus.post(Topic.TASK_CODE, "router", {
        "prompt": "Write a binary search function",
        "model":  "coder",
    })

    # Coder picks it up and posts result
    task = bus.get(Topic.TASK_CODE)
    bus.post(Topic.RESULT_CODE, "backend_agent", {
        "code":    "def binary_search(arr, x): ...",
        "prompt":  task.payload["prompt"],
    }, reply_to=task.id)

    # Reviewer picks up result
    result = bus.get(Topic.RESULT_CODE)
    bus.post(Topic.TASK_REVIEW, "orchestrator", {
        "code":   result.payload["code"],
        "prompt": result.payload["prompt"],
    }, reply_to=result.id)

    # Mark done
    bus.post(Topic.SYSTEM_DONE, "orchestrator", {"status": "complete"})
    console.print("  [green]Pipeline simulation complete.[/green]")

    # 5. Stats + history
    console.print("\n[5] Bus stats:")
    s = bus.stats()
    for k, v in s.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")

    console.print()
    bus.print_history()

    console.rule("[bold green]Bus test done![/bold green]")