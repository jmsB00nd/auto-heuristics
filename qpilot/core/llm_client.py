import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from rich.console import Console

from utils.utils import run_cli_json
from ..config.settings import OrchestratorConfig

console = Console()


class LLMClient:
    """Dedicated client for interacting with the LLM and managing context.

    Token accounting is word-count based: each LLM call's tokens =
    ``len(prompt.split()) + len(response.split())``. ``usage_totals``
    accumulates that count across the whole run. ``call_log`` keeps a
    per-call snapshot (useful for tokens-vs-metrics plots).

    Thread-safety: ``query()`` can be called from multiple threads concurrently.
    Subprocess invocations run without holding any lock (so calls truly run in
    parallel); only the post-call state mutations (usage totals, call log,
    conversation history) are serialized under ``_lock``.
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []

        self.usage_totals: Dict[str, int] = {"total_tokens": 0}
        self.call_log: List[Dict] = []
        self.current_stage: str = "init"
        self._lock = threading.Lock()

    @property
    def total_tokens(self) -> int:
        return int(self.usage_totals["total_tokens"])

    @contextmanager
    def stage(self, name: str):
        prev = self.current_stage
        self.current_stage = name
        try:
            yield
        finally:
            self.current_stage = prev

    def query(self, prompt_text: str, reset_conversation: bool = False, show_counter: Optional[bool] = None) -> Tuple[Optional[str], Optional[Dict]]:
        """Call the LLM. Returns ``(response, call_record)``. ``call_record`` is
        the per-call dict appended to ``call_log`` (snapshots cumulative token
        counts atomically under the same lock as the totals update — caller
        gets the cumulative state at THIS call's commit, not a later moment).
        Both elements are ``None`` if the subprocess fails.

        ``show_counter=None`` defers to the config default; pass
        ``show_counter=False`` explicitly when running many calls in parallel
        to avoid overlapping Live-display widgets on the console."""
        with self._lock:
            if reset_conversation:
                history_snapshot: List[Dict[str, str]] = []
            else:
                history_snapshot = list(self.conversation_history)
            stage_at_call = self.current_stage

        history_snapshot.append({"role": "user", "content": prompt_text})
        formatted_input = self._format_messages(history_snapshot)

        effective_counter = self.config.show_token_counter if show_counter is None else show_counter

        try:
            response, usage = run_cli_json(
                self.config.cli_command,
                formatted_input,
                show_counter=bool(effective_counter),
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error: CLI returned exit code {e.returncode}[/bold red]")
            return None, None
        except Exception as e:
            console.print(f"[bold red]Error: Execution failed: {e}[/bold red]")
            return None, None

        with self._lock:
            self.usage_totals["total_tokens"] += usage.get("total_tokens", 0)

            call_record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage_at_call,
                "call_total_tokens": int(usage.get("total_tokens", 0)),
                "cumulative_total_tokens": self.total_tokens,
            }
            self.call_log.append(call_record)

            if self.config.use_conversation_mode and not reset_conversation:
                self.conversation_history = history_snapshot + [
                    {"role": "assistant", "content": response}
                ]

        return response, call_record

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        if not self.config.use_conversation_mode or len(messages) <= 1:
            return messages[-1]['content']

        formatted = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n\n"
        return formatted.strip()
