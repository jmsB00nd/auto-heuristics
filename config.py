from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OrchestratorConfig:
    cli_command: str
    backend: str
    benchmark_dir: str
    prompts_dir: str
    history_file: str = "experiment_history.json"
    use_conversation_mode: bool = True
    send_context_api: bool = False
    show_token_counter: bool = True
    target_top_ideas: int = 5
    min_score_threshold: int = 6
    run_stage1_literature_review: bool = False
    timeout_seconds: int = 200
    top_ideas_to_implement: int = 5
    
    # Generated at runtime
    current_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    @property
    def log_dir(self) -> str:
        return f"logs/{self.current_time}_run"