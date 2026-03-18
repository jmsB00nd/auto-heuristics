from config import OrchestratorConfig
from orchestrator import OrchestratorV2


config = OrchestratorConfig(
    cli_command=["claude", "--model", "claude-sonnet-4-6", "-p", "--output-format", "text"],
    backend="ibm_sherbrooke",
    benchmark_dir="benchmarks/qasmbench-large/",
    prompts_dir="/home/jmsb00nd/Documents/auto-heuristics/prompts",
    problem="routing"
)

orchestrator = OrchestratorV2(config)
orchestrator.run_full_pipeline()