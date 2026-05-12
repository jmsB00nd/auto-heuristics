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
    target_top_ideas: int = 5
    run_stage1_literature_review: bool = False
    timeout_seconds: int = 200
    top_ideas_to_implement: int = 5
    implementation_workers: int = 4

    # Evolution loop (HD-KG hypothesis-driven reflection + crossover + mutation)
    run_evolution: bool = True
    pop_size: int = 5
    mutation_rate: float = 0.5
    max_fe: int = 5
    evolution_workers: int = 4
    active_memory_limit: int = 20

    # Stagnation-driven exploration restart
    stagnation_patience: int = 3
    stagnation_eps: float = 0.005

    # HD-KG hyperparameters
    kg_alpha: float = 0.2
    kg_confidence_threshold: float = 0.75
    kg_open_sample_prob: float = 0.3

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
