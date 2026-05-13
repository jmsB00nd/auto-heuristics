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


def save_log(stage_dir: str, filename: str, content: str, quiet: bool = False):
    """Save a text log file inside the given stage directory.

    ``quiet=True`` suppresses the console echo — useful in high-volume
    stages (e.g. per-child crossover artifacts) where printing one line
    per file drowns the meaningful output.
    """
    ensure_dir(stage_dir)
    filepath = os.path.join(stage_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    if not quiet:
        console.print(f"[dim]  -> Saved {filepath}[/dim]")
    return filepath


def save_json(stage_dir: str, filename: str, data, quiet: bool = False):
    """Save a JSON log file inside the given stage directory.

    See ``save_log`` for ``quiet``.
    """
    ensure_dir(stage_dir)
    filepath = os.path.join(stage_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    if not quiet:
        console.print(f"[dim]  -> Saved {filepath}[/dim]")
    return filepath


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def file_to_string(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        return file.read()


def _empty_usage() -> dict:
    return {"total_tokens": 0}


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
    text.append("total: ", style="white")
    text.append(f"{usage['total_tokens']:,}", style="bold magenta")
    text.append(" words", style="dim")
    return text


def run_cli_json(command: str, input_text: str, show_counter: bool = True):
    """Run the Claude CLI with ``--output-format json`` and return (text, usage).

    Tokens are word-counted (whitespace split) over prompt + response.
    The CLI's billed-token envelope is parsed only enough to recover the
    response text; its ``usage`` numbers are not used.
    ``usage`` is a single-key dict: ``{"total_tokens": int}``.
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
        # Fallback: the CLI printed something that isn't JSON — count what we
        # sent in so the call isn't invisible to the running totals.
        usage = {"total_tokens": len(input_text.split())}
        if show_counter:
            console.print("[yellow]Warning: CLI did not return JSON; counting prompt words only.[/yellow]")
        return stdout.strip(), usage

    result_text = payload.get("result")
    if result_text is None:
        # Some CLI modes put assistant text under "content" or nested shapes.
        result_text = payload.get("content") or ""
    result_text = result_text.strip() if isinstance(result_text, str) else json.dumps(result_text)

    usage = {"total_tokens": len(input_text.split()) + len(result_text.split())}

    if show_counter:
        console.print(_final_display(usage))

    return result_text, usage


def run_with_token_counter(command, input_text):
    text, usage = run_cli_json(command, input_text, show_counter=True)
    return text, usage["total_tokens"]