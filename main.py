from config import OrchestratorConfig
from orchestrator import OrchestratorV2


config = OrchestratorConfig(
    cli_command=["claude", "--model", "claude-opus-4-6", "-p", "--output-format", "text"],
    backend="ibm_sherbrooke",
    benchmark_dir="benchmarks/queko-bss-16qbt/",
    prompts_dir="/home/jmsb00nd/Documents/auto-heuristics/prompts",
    problem="mapping"
)

orchestrator = OrchestratorV2(config)
orchestrator.run_full_pipeline()
