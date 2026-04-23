import os
import json
import time
from datetime import datetime
import subprocess
from rich.console import Console
from rich.text import Text
from rich.live import Live

console = Console()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_log(stage_dir: str, filename: str, content: str):
    """Save a text log file inside the given stage directory."""
    ensure_dir(stage_dir)
    filepath = os.path.join(stage_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[dim]  -> Saved {filepath}[/dim]")
    return filepath


def save_json(stage_dir: str, filename: str, data):
    """Save a JSON log file inside the given stage directory."""
    ensure_dir(stage_dir)
    filepath = os.path.join(stage_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"[dim]  -> Saved {filepath}[/dim]")
    return filepath


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def file_to_string(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        return file.read()


_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _empty_usage() -> dict:
    return {k: 0 for k in _USAGE_KEYS} | {"total_cost_usd": 0.0}


def _busy_display(elapsed: float) -> Text:
    text = Text()
    text.append("🤖 ", style="bold green")
    text.append("Querying LLM... ", style="bold cyan")
    text.append(f"{elapsed:0.1f}s", style="bold yellow")
    return text


def _final_display(usage: dict) -> Text:
    text = Text()
    text.append("✓ ", style="bold green")
    text.append("LLM call complete — ", style="cyan")
    text.append("in: ", style="white")
    text.append(f"{usage['input_tokens']:,}", style="bold yellow")
    text.append("  out: ", style="white")
    text.append(f"{usage['output_tokens']:,}", style="bold yellow")
    text.append("  total: ", style="white")
    text.append(f"{usage['input_tokens'] + usage['output_tokens']:,}", style="bold magenta")
    text.append(f"  (${usage['total_cost_usd']:.4f})", style="dim")
    return text


def run_cli_json(command: str, input_text: str, show_counter: bool = True):
    """Run the Claude CLI with ``--output-format json`` and return (text, usage).

    The CLI's JSON envelope has shape::

        {"type": "result", "result": "...", "usage": {...}, "total_cost_usd": 0.012}

    ``usage`` dict is normalized to always contain the four token fields
    plus ``total_cost_usd``. Missing keys default to 0.
    """
    start = time.time()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        shell=True,
    )

    def _wait_and_collect():
        out, err = process.communicate(input=input_text)
        return out, err

    if show_counter:
        process.stdin.write(input_text)
        process.stdin.close()
        with Live(_busy_display(0.0), refresh_per_second=4, console=console) as live:
            while process.poll() is None:
                live.update(_busy_display(time.time() - start))
                time.sleep(0.25)
            stdout = process.stdout.read()
            stderr = process.stderr.read()
    else:
        stdout, stderr = _wait_and_collect()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, stderr)

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        # Fallback: the CLI printed something that isn't JSON — surface raw stdout
        # and zeroed usage rather than crashing the whole run.
        usage = _empty_usage()
        if show_counter:
            console.print("[yellow]Warning: CLI did not return JSON; usage unknown.[/yellow]")
        return stdout.strip(), usage

    raw_usage = payload.get("usage", {}) or {}
    usage = {k: int(raw_usage.get(k, 0) or 0) for k in _USAGE_KEYS}
    usage["total_cost_usd"] = float(payload.get("total_cost_usd", 0.0) or 0.0)

    result_text = payload.get("result")
    if result_text is None:
        # Some CLI modes put assistant text under "content" or nested shapes.
        result_text = payload.get("content") or ""
    result_text = result_text.strip() if isinstance(result_text, str) else json.dumps(result_text)

    if show_counter:
        console.print(_final_display(usage))

    return result_text, usage


# Back-compat shim: old call sites expected (text, token_count). Token count here
# is (input + output) from the authoritative CLI usage report.
def run_with_token_counter(command, input_text):
    text, usage = run_cli_json(command, input_text, show_counter=True)
    return text, usage["input_tokens"] + usage["output_tokens"]