import os
import json
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
    
    
def create_counter_display(tokens, chars):
    text = Text()
    text.append("🤖 ", style="bold green")
    text.append("LLM Generating... ", style="bold cyan")
    text.append(f"Tokens: ", style="white")
    text.append(f"{tokens:,}", style="bold yellow")
    text.append(" | ", style="dim")
    text.append(f"Characters: ", style="white")
    text.append(f"{chars:,}", style="bold blue")
    return text
    
    
def run_with_token_counter(command, input_text):
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        shell=True,
        bufsize=1,
    )
    process.stdin.write(input_text)
    process.stdin.close()

    full_response = ""
    token_count = 0

    with Live(create_counter_display(0, 0), refresh_per_second=10, console=console) as live:
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            full_response += char
            if char in ' \n\t.,;:!?':
                token_count += 1
            if len(full_response) % 10 == 0:
                live.update(create_counter_display(token_count, len(full_response)))
        live.update(create_counter_display(token_count, len(full_response)))

    process.wait()
    if process.returncode != 0:
        stderr = process.stderr.read()
        raise subprocess.CalledProcessError(process.returncode, command, stderr)
    
    # Return both response text and counted output tokens to the caller
    return full_response.strip(), token_count