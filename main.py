import argparse

from agent.config import OrchestratorConfig
from agent.qpilot import Qpilot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the auto-heuristics pipeline end-to-end."
    )
    parser.add_argument(
        "--experiment-name",
        default="run",
        help="Slug appended to the run_id (default: 'run' — preserves legacy "
        "'{timestamp}_run' log directory naming).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Python / NumPy RNGs. Recorded in run_metadata.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = OrchestratorConfig(
        cli_command="claude --model claude-opus-4-7 -p --output-format json",
        backend="ibm_sherbrooke",
        benchmark_dir="benchmarks/qasmbench-large/",
        prompts_dir="/home/jmsb00nd/Documents/auto-heuristics/prompts",
        problem="mapping",
        run_stage3_5_reflection=True,
        run_stage4_iterative_refinement=True,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )

    qpilot = Qpilot(config)
    qpilot.run()


if __name__ == "__main__":
    main()
