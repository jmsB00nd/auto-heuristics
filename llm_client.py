import subprocess
from typing import List, Dict, Optional
from rich.console import Console
from utils.utils import run_with_token_counter
from config import OrchestratorConfig

console = Console()

class LLMClient:
    """Dedicated client for interacting with the LLM and managing context."""
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        self.total_tokens: int = 0

    def query(self, prompt_text: str, reset_conversation: bool = False) -> Optional[str]:
        if reset_conversation:
            self.conversation_history = []

        self.conversation_history.append({"role": "user", "content": prompt_text})
        formatted_input = self._format_conversation()

        try:
            if self.config.show_token_counter:
                response, token_count = run_with_token_counter(self.config.cli_command, formatted_input)
                self.total_tokens += token_count
            else:
                with console.status("[bold green]Querying LLM...", spinner="dots"):
                    result = subprocess.run(
                        self.config.cli_command,
                        input=formatted_input,
                        capture_output=True,
                        text=True,
                        check=True,
                        encoding='utf-8',
                        shell=True,
                    )
                    response = result.stdout.strip()
                    self.total_tokens += len(response.split())

            if self.config.use_conversation_mode:
                self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error: CLI returned exit code {e.returncode}[/bold red]")
            return None
        except Exception as e:
            console.print(f"[bold red]Error: Execution failed: {e}[/bold red]")
            return None

    def _format_conversation(self) -> str:
        if not self.config.use_conversation_mode or len(self.conversation_history) <= 1:
            return self.conversation_history[-1]['content']
            
        formatted = ""
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n\n"
        return formatted.strip()