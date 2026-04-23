import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return slug or "run"


@dataclass
class OrchestratorConfig:
    cli_command: str
    backend: str
    benchmark_dir: str
    prompts_dir: str
    problem: str
    history_file: str = "outputs/experiment_history.json"
    use_conversation_mode: bool = True
    send_context_api: bool = False
    show_token_counter: bool = True
    target_top_ideas: int = 2
    run_stage1_literature_review: bool = False
    timeout_seconds: int = 200
    top_ideas_to_implement: int = 2
    implementation_workers: int = 4

    # New Pipeline Toggles
    run_stage3_5_reflection: bool = True
    run_stage4_iterative_refinement: bool = True
    refinement_rounds: int = 1
    crossover_rate: float = 0.5
    diversity_pool_size: int = 5
    stagnation_threshold: int = 3
    active_memory_limit: int = 20

    # Per-run plotting of cumulative tokens vs. heuristic quality metrics.
    generate_plots: bool = True

    # Reproducibility / run identity
    experiment_name: str = "run"
    seed: Optional[int] = None

    # Generated at runtime
    current_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    @property
    def run_id(self) -> str:
        return f"{self.current_time}_{_slugify(self.experiment_name)}"

    @property
    def log_dir(self) -> str:
        return f"outputs/logs/{self.run_id}"
