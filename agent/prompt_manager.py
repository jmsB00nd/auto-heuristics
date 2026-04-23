from pathlib import Path
from rich.console import Console
from utils.utils import file_to_string

console = Console()

class PromptManager:
    """Handles loading and formatting of prompt templates."""
    def __init__(self, prompts_dir: str, problem: str):
        self.prompts_dir = Path(prompts_dir)
        self.problem = problem
        self.system_generator = self._load(f"{self.problem}/system_generator.txt")
        self.output_format = self._load(f"{self.problem}/output_format.txt")
        self.idea_prompt = self._load(f"{self.problem}/ideas_generation.txt")
        self.code = self._load(f"{self.problem}/code.txt")
        self.variables = self._load(f"{self.problem}/variables.txt")
        self.lit_review_prompt = self._load(f"{self.problem}/literature_review.txt")
        self.refinement_prompt = self._load(f"{self.problem}/refinement.txt")
        self.memory_summary_prompt = self._load(f"{self.problem}/memory_summary.txt")
        self.crossover_prompt = self._load(f"{self.problem}/crossover.txt")
        self.reflection_prompt = self._load(f"{self.problem}/reflection.txt")

    def _load(self, relative_path: str) -> str:
        path = self.prompts_dir / relative_path
        try:
            return file_to_string(str(path))
        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not load prompt {path}: {e}[/bold yellow]")
            return ""