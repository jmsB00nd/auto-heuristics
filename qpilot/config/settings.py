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
    target_top_ideas: int = 10
    run_stage1_literature_review: bool = False
    timeout_seconds: int = 200
    top_ideas_to_implement: int = 10
    implementation_workers: int = 2

    run_evolution: bool = True
    pop_size: int = 10
    crossover_count: int = 4
    max_fe: int = 50

    max_time_seconds: Optional[float] = None
    evolution_workers: int = 2
    active_memory_limit: int = 20
 
    use_kg: bool = True
    use_crossover: bool = True
    use_mutation: bool = True
    use_reideation: bool = True
    use_ideation_memory: bool = True

    kg_alpha: float = 0.2
    kg_confidence_threshold: float = 0.6
    kg_open_sample_prob: float = 0.5

    generate_plots: bool = True

    experiment_name: str = "run"
    seed: Optional[int] = None

    current_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    @property
    def run_id(self) -> str:
        return f"{self.current_time}_{_slugify(self.experiment_name)}"

    @property
    def log_dir(self) -> str:
        return f"outputs/logs/{self.run_id}"
