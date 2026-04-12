from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OrchestratorConfig:
    cli_command: str
    backend: str
    benchmark_dir: str
    prompts_dir: str
    problem: str
    history_file: str = "experiment_history.json"
    use_conversation_mode: bool = True
    send_context_api: bool = False
    show_token_counter: bool = True
    target_top_ideas: int = 5
    min_score_threshold: int = 5
    run_stage1_literature_review: bool = False
    timeout_seconds: int = 200
    top_ideas_to_implement: int = 5
    
    # New Pipeline Toggles
    run_stage3_5_reflection: bool = True
    run_stage4_iterative_refinement: bool = True
    refinement_rounds: int = 10
    crossover_rate: float = 0.5
    diversity_pool_size: int = 5
    stagnation_threshold: int = 3
    active_memory_limit: int = 20

    # Generated at runtime
    current_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    @property
    def log_dir(self) -> str:
        return f"logs/{self.current_time}_run"