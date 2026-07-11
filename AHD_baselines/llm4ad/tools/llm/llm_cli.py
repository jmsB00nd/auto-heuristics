"""
CLI-based LLM interface for Claude and Gemini command-line tools.

Supports:
  - Claude CLI: ["claude", "--model", "claude-sonnet-4-6", "-p", "--output-format", "text"]
  - Gemini CLI: ["gemini", "-m", "gemini-2.5-flash"]
  - npx Gemini: ["npx", "gemini", "-m", "gemini-3-flash-preview"]

Inherits from the LLM4AD base LLM class so it can be used as a drop-in
replacement for HttpsApi wherever the framework expects an LLM instance.
"""

from __future__ import annotations

import json as _json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List

from ...base import LLM

try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text

    console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    console = None


class CliLLM(LLM):
    """LLM backend that shells out to a CLI tool (Claude CLI or Gemini CLI).

    Parameters
    ----------
    cli_command : list[str]
        The command + arguments used to invoke the CLI.
        Examples:
            ["claude", "--model", "claude-sonnet-4-6", "-p", "--output-format", "text"]
            ["gemini", "-m", "gemini-2.5-flash"]
            ["npx", "gemini", "-m", "gemini-3-flash-preview"]
    use_conversation_mode : bool
        When True, the full conversation history is sent on every call so the
        CLI tool can produce contextual replies.  **Note:** The LLM4AD framework
        already sends self-contained prompts with all necessary context, so this
        should typically be False (the default).
    show_token_counter : bool
        When True and *rich* is installed, a live token/character counter is
        displayed while the CLI streams its response.
    timeout : int | None
        Optional timeout (seconds) for the subprocess.  ``None`` means wait
        indefinitely.
    """

    def __init__(
        self,
        cli_command: list[str],
        use_conversation_mode: bool = False,
        show_token_counter: bool = True,
        timeout: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Accept either a pre-split list or a plain string from the GUI entry.
        # IMPORTANT: list("claude --model...") splits char-by-char → ['c','l','a',...]
        # which makes subprocess run 'c' as the command.  Use shlex instead.
        if isinstance(cli_command, str):
            # On Windows shlex.split can misbehave with backslashes;
            # a simple whitespace split is safer for typical CLI strings.
            if sys.platform == "win32":
                self.cli_command = cli_command.split()
            else:
                self.cli_command = shlex.split(cli_command)
        else:
            self.cli_command = list(cli_command)

        # Resolve the executable via PATH so we can run without shell=True.
        # shell=True on Windows wraps in cmd.exe /c which swallows stdin,
        # causing "Input must be provided" errors from the Claude CLI.
        resolved = shutil.which(self.cli_command[0])
        if resolved:
            self.cli_command[0] = resolved

        self.use_conversation_mode = use_conversation_mode
        self.show_token_counter = show_token_counter and _HAS_RICH
        self.timeout = timeout
        self.conversation_history: list[dict[str, str]] = []
        self._lock = threading.Lock()
        # Token tracking for last LLM call
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_thinking_tokens: int = 0

        # Cumulative usage across the run — mirrors agent.llm_client.LLMClient.
        # Authoritative totals come from the Claude CLI stream-json envelope
        # (same numbers billed by the API). For non-Claude CLIs cache fields
        # and cost stay at 0 and token counts are character-based estimates.
        self.usage_totals: Dict[str, float] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "thinking_tokens": 0,
            "total_cost_usd": 0.0,
        }
        # Per-call snapshot (each entry carries cumulative state at commit time).
        self.call_log: List[Dict[str, Any]] = []
        self.current_stage: str = "init"

        # Extract model name from CLI command for compatibility with external LLM libraries
        self.model = self._extract_model_name()

    @property
    def total_tokens(self) -> int:
        return int(self.usage_totals["input_tokens"] + self.usage_totals["output_tokens"])

    @contextmanager
    def stage(self, name: str):
        """Tag subsequent calls with a stage name (recorded in ``call_log``)."""
        prev = self.current_stage
        self.current_stage = name
        try:
            yield
        finally:
            self.current_stage = prev

    def _record_usage(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Update cumulative totals + append a call record. Returns the record.

        Mirrors ``agent.llm_client.LLMClient.query`` — snapshots cumulative
        state atomically under the same lock as the totals update so the
        caller sees the cumulative state at THIS call's commit, not a later
        moment. Also keeps the ``last_*`` attributes the samplers read in
        sync (still under the lock to avoid torn reads).
        """
        with self._lock:
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "thinking_tokens",
            ):
                self.usage_totals[key] += int(usage.get(key, 0) or 0)
            self.usage_totals["total_cost_usd"] += float(usage.get("total_cost_usd", 0.0) or 0.0)

            self.last_prompt_tokens = int(usage.get("input_tokens", 0) or 0)
            self.last_completion_tokens = int(usage.get("output_tokens", 0) or 0)
            self.last_thinking_tokens = int(usage.get("thinking_tokens", 0) or 0)

            call_record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": self.current_stage,
                "call_input_tokens": self.last_prompt_tokens,
                "call_output_tokens": self.last_completion_tokens,
                "call_thinking_tokens": self.last_thinking_tokens,
                "call_cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
                "call_cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
                "call_cost_usd": float(usage.get("total_cost_usd", 0.0) or 0.0),
                "cumulative_input_tokens": int(self.usage_totals["input_tokens"]),
                "cumulative_output_tokens": int(self.usage_totals["output_tokens"]),
                "cumulative_total_tokens": self.total_tokens,
                "cumulative_cost_usd": float(self.usage_totals["total_cost_usd"]),
            }
            self.call_log.append(call_record)
        return call_record

    def __getstate__(self) -> dict:
        """Prepare object for pickling by removing non-picklable objects.
        
        Called by pickle/joblib when serializing this object for multiprocessing.
        Excludes non-picklable attributes like thread locks and logger instances.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable thread lock
        state.pop('_lock', None)
        # Remove logger if present (loggers often contain non-picklable streams/handlers)
        state.pop('logger', None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore object from pickled state and recreate non-picklable objects.
        
        Called by pickle/joblib when deserializing this object from multiprocessing.
        Recreates the _lock and logger that were removed during pickling.
        """
        self.__dict__.update(state)
        # Recreate the thread lock
        self._lock = threading.Lock()
        # Logger will be set again if needed via set_logger() method
        if not hasattr(self, 'logger'):
            self.logger = None

    def _extract_model_name(self) -> str:
        """Extract the model name from the CLI command.
        
        Handles:
          - Claude: ["claude", "--model", "claude-sonnet-4-6", ...]
          - Gemini: ["gemini", "-m", "gemini-2.5-flash", ...] or
                    ["npx", "gemini", "-m", "gemini-3-flash-preview"]
        
        Returns
        -------
        str
            The extracted model name, or a generic name if not found.
        """
        try:
            # Look for --model flag (Claude style)
            if "--model" in self.cli_command:
                idx = self.cli_command.index("--model")
                if idx + 1 < len(self.cli_command):
                    return self.cli_command[idx + 1]
            
            # Look for -m flag (Gemini style)
            if "-m" in self.cli_command:
                idx = self.cli_command.index("-m")
                if idx + 1 < len(self.cli_command):
                    return self.cli_command[idx + 1]
            
            # Fallback: use the first argument (executable name)
            return self.cli_command[0] if self.cli_command else "unknown"
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Public API  (LLM interface)
    # ------------------------------------------------------------------

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """Send *prompt* to the CLI and return the response text.

        *prompt* may be a plain string **or** a list of message dicts
        (``[{"role": "user", "content": "..."}, ...]``) for compatibility
        with code that was written for the HTTPS API.

        Some methods (e.g. PartEvo) instead pass ``prompt=""`` and put the
        real payload in a ``messages=`` keyword argument.  Fall back to that
        if the positional prompt is empty.

        Each call is **independent** — the framework provides self-contained
        prompts so no cross-call context is needed.
        """
        if isinstance(prompt, list):
            # Convert chat-style messages into a single text block
            text = self._messages_to_text(prompt)
        else:
            text = str(prompt) if prompt is not None else ""

        # If the positional prompt is empty, look for messages in kwargs.
        if not text.strip():
            messages = kwargs.get("messages")
            if isinstance(messages, list) and messages:
                text = self._messages_to_text(messages)

        # Guard against sending an empty prompt — the Claude/Gemini CLIs
        # both error out with "Input must be provided" when stdin is blank.
        if not text.strip():
            if self.debug_mode:
                print("[CliLLM] draw_sample called with empty prompt and no messages; returning empty string.")
            return ""

        result = self.query(text)
        # Never return None — downstream code (SampleTrimmer, EoHSampler) expects a string.
        return result if result is not None else ""

    @staticmethod
    def _messages_to_text(messages: list[dict[str, Any]]) -> str:
        """Flatten a list of ``{"role": ..., "content": ...}`` dicts to plain text.

        Handles three content shapes:
          - ``content`` is a plain string (HTTPS API style).
          - ``message`` is a plain string (EoHPrompt ``create_instruct_prompt`` style).
          - ``content`` is a list of multimodal parts (PartEvoPrompt style),
            e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``.
            Text parts are concatenated; image parts are replaced with a placeholder.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            # Support both 'content' (HTTPS API style) and 'message' (EoHPrompt style)
            body = msg.get("content")
            if body is None:
                body = msg.get("message", "")

            if isinstance(body, list):
                # Multimodal content — flatten text parts, placeholder for images.
                chunks = []
                for item in body:
                    if not isinstance(item, dict):
                        chunks.append(str(item))
                        continue
                    itype = item.get("type")
                    if itype == "text":
                        chunks.append(str(item.get("text", "")))
                    elif itype == "image_url":
                        chunks.append("[image omitted]")
                    else:
                        # Unknown part type — fall back to any 'text' field.
                        if "text" in item:
                            chunks.append(str(item["text"]))
                body_text = "\n".join(c for c in chunks if c)
            else:
                body_text = str(body)

            parts.append(f"{role}: {body_text}")
        return "\n\n".join(parts)

    def draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]:
        return [self.draw_sample(p, *args, **kwargs) for p in prompts]

    def set_logger(self, logger: Any) -> None:
        """Set a logger instance for compatibility with external LLM libraries.
        
        The 'llamea' library and other external LLM frameworks may call this
        method to provide a logger for debug/info messages. This is a no-op
        for CLI-based LLMs, but the method must exist to satisfy the interface.
        
        Parameters
        ----------
        logger : Any
            A logger instance (typically a Python logging.Logger).
        """
        self.logger = logger

    # ------------------------------------------------------------------
    # Core query logic  (mirrors OrchestratorV2.query_claude)
    # ------------------------------------------------------------------

    def query(self, prompt_text: str, reset_conversation: bool = False) -> str | None:
        """Query the LLM via the CLI, optionally maintaining conversation history.

        Parameters
        ----------
        prompt_text : str
            The user prompt to send.
        reset_conversation : bool
            If True the conversation history is cleared before this turn.

        Returns
        -------
        str or None
            The model's response, or None on failure.
        """
        try:
            with self._lock:
                if reset_conversation:
                    self.conversation_history = []

                # Build the text that will be piped into the CLI
                if self.use_conversation_mode:
                    self.conversation_history.append({"role": "user", "content": prompt_text})
                    if len(self.conversation_history) > 1:
                        formatted_input = self._format_conversation_for_cli()
                    else:
                        formatted_input = prompt_text
                else:
                    # Stateless mode (default): each call is independent,
                    # matching the behaviour of HttpsApi.draw_sample.
                    formatted_input = prompt_text

            if self.debug_mode:
                print(f"\n{'='*60}")
                print(f"[CliLLM] Sending prompt ({len(formatted_input)} chars):")
                print(formatted_input[:500] + ("..." if len(formatted_input) > 500 else ""))
                print(f"{'='*60}")

            # Execute
            if self.show_token_counter:
                response = self._run_with_token_counter(formatted_input)
            else:
                response = self._run_simple(formatted_input)

            if self.debug_mode:
                print(f"\n{'-'*60}")
                print(f"[CliLLM] Response ({len(response) if response else 0} chars):")
                print((response[:500] + "...") if response and len(response) > 500 else response)
                print(f"{'-'*60}")

            if self.use_conversation_mode and response:
                with self._lock:
                    self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except subprocess.CalledProcessError as e:
            errmsg = f"CLI returned exit code {e.returncode}"
            stderr_out = getattr(e, "stderr", "") or getattr(e, "output", "") or ""
            if stderr_out:
                errmsg += f"\nstderr: {stderr_out[:500]}"
            if self.debug_mode:
                print(f"[CliLLM ERROR] {errmsg}")
            if _HAS_RICH:
                console.print(f"[bold red]Error:[/bold red] {errmsg}")
            else:
                print(f"[CliLLM ERROR] {errmsg}")
            return None
        except subprocess.TimeoutExpired as e:
            errmsg = f"CLI timed out after {self.timeout}s"
            if self.debug_mode:
                print(f"[CliLLM ERROR] {errmsg}")
            if _HAS_RICH:
                console.print(f"[bold red]Error:[/bold red] {errmsg}")
            else:
                print(f"[CliLLM ERROR] {errmsg}")
            return None
        except Exception as e:
            errmsg = f"Execution failed: {e}"
            if self.debug_mode:
                print(f"[CliLLM ERROR] {errmsg}")
            if _HAS_RICH:
                console.print(f"[bold red]Error:[/bold red] {errmsg}")
            else:
                print(f"[CliLLM ERROR] {errmsg}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_claude_cli(self) -> bool:
        """Return True when the CLI executable is the Claude CLI."""
        if not self.cli_command:
            return False
        return 'claude' in os.path.basename(self.cli_command[0]).lower()

    def _build_json_command(self) -> list[str]:
        """Return a copy of the CLI command that requests stream-json output.

        stream-json emits one JSON object per line, including a full assistant
        message event that contains thinking content blocks.  This is the only
        output format that exposes thinking text so we can count thinking tokens.
        """
        cmd = list(self.cli_command)
        if '--output-format' in cmd:
            idx = cmd.index('--output-format')
            if idx + 1 < len(cmd):
                cmd[idx + 1] = 'stream-json'
        else:
            cmd.extend(['--output-format', 'stream-json'])
        return cmd

    def _parse_json_response(self, stdout: str) -> tuple[str, Dict[str, Any]]:
        """Parse Claude CLI stream-json output (one JSON object per line).

        Returns ``(text, usage)`` where ``usage`` carries the authoritative
        numbers billed by the API:

        - ``input_tokens`` / ``output_tokens``
        - ``cache_creation_input_tokens`` / ``cache_read_input_tokens``
        - ``thinking_tokens`` (estimated from thinking text @ ~3.5 chars/token)
        - ``total_cost_usd``

        The stream-json format emits an 'assistant' event with full content
        blocks (including thinking when extended thinking is active) and a
        final 'result' event whose usage/cost is the most authoritative.
        ``output_tokens`` is the exact API value and already includes any
        thinking tokens.

        On a parse failure returns ``(stdout, {})``.
        """
        text = ''
        usage: Dict[str, Any] = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
            'thinking_tokens': 0,
            'total_cost_usd': 0.0,
        }
        thinking_chars = 0

        def _merge_usage(u: Dict[str, Any]) -> None:
            # Later events (esp. 'result') override earlier ones for the same key.
            for k in (
                'input_tokens',
                'output_tokens',
                'cache_creation_input_tokens',
                'cache_read_input_tokens',
            ):
                v = u.get(k)
                if v is not None:
                    try:
                        usage[k] = int(v)
                    except (TypeError, ValueError):
                        pass

        try:
            for raw_line in stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = _json.loads(line)
                except Exception:
                    continue

                etype = event.get('type', '')

                if etype == 'assistant':
                    msg = event.get('message', {}) or {}
                    for block in (msg.get('content') or []):
                        btype = block.get('type', '')
                        if btype == 'thinking':
                            thinking_chars += len(block.get('thinking', '') or '')
                        elif btype == 'text' and not text:
                            text = block.get('text', '') or ''
                    _merge_usage(msg.get('usage', {}) or {})

                elif etype == 'result':
                    _merge_usage(event.get('usage', {}) or {})
                    cost = event.get('total_cost_usd')
                    if cost is not None:
                        try:
                            usage['total_cost_usd'] = float(cost)
                        except (TypeError, ValueError):
                            pass
                    if not text:
                        text = event.get('result', '') or ''

            # ~3.5 chars/token is a reasonable approximation for prose/code thinking.
            usage['thinking_tokens'] = round(thinking_chars / 3.5) if thinking_chars else 0
            return text, usage

        except Exception:
            return stdout, {}

    def _run_simple(self, input_text: str) -> str:
        """Run the CLI synchronously and return the response text.

        For the Claude CLI, requests JSON output so we can extract real
        input/output/thinking token counts from the API response metadata.
        For other CLIs (Gemini, etc.) we fall back to a character-based
        token estimate (~4 chars per token).
        """
        use_json = self._is_claude_cli()
        cmd = self._build_json_command() if use_json else self.cli_command

        def _run(run_cmd, shell=False):
            return subprocess.run(
                run_cmd,
                input=input_text,
                capture_output=True,
                text=True,
                check=True,
                shell=shell,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )

        def _apply_tokens(raw: str) -> str:
            if use_json:
                text, usage = self._parse_json_response(raw)
                self._record_usage(usage)
                return text
            # Non-Claude CLI: char-based estimate, no cache/cost data.
            self._record_usage({
                'input_tokens': len(input_text) // 4,
                'output_tokens': len(raw) // 4,
            })
            return raw

        try:
            if _HAS_RICH:
                with console.status("[bold green]Querying LLM...", spinner="dots"):
                    result = _run(cmd)
            else:
                result = _run(cmd)
            return _apply_tokens(result.stdout.strip())

        except subprocess.CalledProcessError as e:
            stderr_str = getattr(e, "stderr", "") or ""
            if ("stdin" in stderr_str.lower() or "input" in stderr_str.lower()) and "-p" in self.cli_command:
                if self.debug_mode:
                    print("[CliLLM] Stdin error with -p flag. Trying without -p flag...")
                try:
                    fallback_cmd = [arg for arg in cmd if arg != "-p"]
                    cmd_str = " ".join(fallback_cmd)
                    if _HAS_RICH:
                        with console.status("[bold green]Querying LLM (no -p flag)...", spinner="dots"):
                            result = _run(cmd_str, shell=True)
                    else:
                        result = _run(cmd_str, shell=True)
                    if self.debug_mode:
                        print("[CliLLM] Success! Removing -p flag works around the issue.")
                    return _apply_tokens(result.stdout.strip())
                except Exception as retry_err:
                    if self.debug_mode:
                        print(f"[CliLLM] Retry without -p also failed: {retry_err}")
                    raise e
            else:
                raise

    def _run_with_token_counter(self, input_text: str) -> str:
        """Run the CLI in a background thread with a live elapsed-time spinner.

        For the Claude CLI, uses JSON output mode to read exact token counts
        (input, output, and thinking) from the API response metadata.
        For other CLIs, falls back to a character-based estimate (~4 chars/token).
        """
        use_json = self._is_claude_cli()
        cmd = self._build_json_command() if use_json else self.cli_command

        result_container: dict[str, Any] = {"stdout": "", "stderr": "", "error": None}

        def _subprocess_worker():
            try:
                # On Windows the CLI ships as a .CMD wrapper that only receives
                # stdin when invoked through the shell.
                if sys.platform == "win32":
                    proc = subprocess.run(
                        " ".join(cmd),
                        input=input_text,
                        capture_output=True,
                        text=True,
                        check=True,
                        shell=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self.timeout,
                    )
                else:
                    proc = subprocess.run(
                        cmd,
                        input=input_text,
                        capture_output=True,
                        text=True,
                        check=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self.timeout,
                    )
                result_container["stdout"] = proc.stdout.strip()
                result_container["stderr"] = proc.stderr
            except Exception as exc:
                result_container["error"] = exc

        thread = threading.Thread(target=_subprocess_worker, daemon=True)
        thread.start()

        start = time.monotonic()

        live_already_active = getattr(console, "_live", None) is not None

        if live_already_active:
            thread.join()
        else:
            try:
                with Live(
                    self._create_elapsed_display(0.0),
                    refresh_per_second=4,
                    console=console,
                    transient=True,
                ) as live:
                    while thread.is_alive():
                        elapsed = time.monotonic() - start
                        live.update(self._create_elapsed_display(elapsed))
                        thread.join(timeout=0.25)
            except Exception:
                thread.join()

        if result_container["error"] is not None:
            raise result_container["error"]

        raw = result_container["stdout"]
        elapsed = time.monotonic() - start

        if use_json:
            text, usage = self._parse_json_response(raw)
            self._record_usage(usage)
            if _HAS_RICH:
                inp = self.last_prompt_tokens
                out = self.last_completion_tokens
                think = self.last_thinking_tokens
                cache_r = int(usage.get('cache_read_input_tokens', 0) or 0)
                cache_c = int(usage.get('cache_creation_input_tokens', 0) or 0)
                cost = float(usage.get('total_cost_usd', 0.0) or 0.0)
                think_str = f" (think: {think:,})" if think else ""
                cache_str = f" | cache r/c: {cache_r:,}/{cache_c:,}" if (cache_r or cache_c) else ""
                cost_str = f" | ${cost:.4f}" if cost else ""
                console.print(
                    f"[bold green]✓[/bold green] LLM responded — "
                    f"in: {inp:,} | out: {out:,}{think_str}{cache_str}{cost_str}, "
                    f"{len(text):,} chars, {elapsed:.1f}s"
                )
        else:
            text = raw
            self._record_usage({
                'input_tokens': len(input_text) // 4,
                'output_tokens': len(text) // 4,
            })
            if _HAS_RICH:
                console.print(
                    f"[bold green]✓[/bold green] LLM responded — "
                    f"~{self.last_completion_tokens:,} tokens (estimated), "
                    f"{len(text):,} chars, {elapsed:.1f}s"
                )
        return text

    # ---- Rich display helpers ----------------------------------------

    @staticmethod
    def _create_elapsed_display(elapsed: float) -> Text:
        """Spinner-style display while waiting for the subprocess."""
        text = Text()
        text.append("🤖 ", style="bold green")
        text.append("LLM Generating... ", style="bold cyan")
        text.append(f"{elapsed:.1f}s", style="bold yellow")
        return text

    @staticmethod
    def _create_final_display(input_tokens: int, output_tokens: int, thinking_tokens: int, chars: int, elapsed: float) -> Text:
        """Final display with exact input/output/thinking token counts."""
        text = Text()
        text.append("🤖 ", style="bold green")
        text.append("LLM Done ", style="bold cyan")
        text.append(f"in: {input_tokens:,} | out: {output_tokens:,}", style="bold yellow")
        if thinking_tokens:
            text.append(f" (think: {thinking_tokens:,})", style="bold magenta")
        text.append(" | ", style="dim")
        text.append(f"{chars:,} chars", style="bold blue")
        text.append(" | ", style="dim")
        text.append(f"{elapsed:.1f}s", style="bold green")
        return text

    def _format_conversation_for_cli(self) -> str:
        formatted = ""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n\n"
            else:
                formatted += f"Assistant: {msg['content']}\n\n"
        return formatted.strip()

    def close(self):
        """Nothing to clean up for a CLI backend."""
        pass

