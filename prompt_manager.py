from pathlib import Path
from rich.console import Console
from utils.utils import file_to_string

console = Console()

class PromptManager:
    """Handles loading and formatting of prompt templates."""
    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        self.baseline = self._load("routing/baseline.txt")
        self.system_generator = self._load("common/system_generator.txt")
        self.documentation = self._load("common/documentation.txt")
        self.output_format = self._load("routing/output_format.txt")
        self.idea_prompt = self._load("routing/ideas_generation.txt")
        self.idea_history = self._load("routing/ideas_history.txt")
        self.lit_review_prompt = self._load("routing/literature_review.txt")

    def _load(self, relative_path: str) -> str:
        path = self.prompts_dir / relative_path
        try:
            return file_to_string(str(path))
        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not load prompt {path}: {e}[/bold yellow]")
            return ""