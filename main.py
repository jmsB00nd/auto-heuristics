import argparse
import re

from qpilot import OrchestratorConfig, Qpilot


def _parse_duration(s: str) -> float:
    """Parse a wall-clock duration to seconds. Accepts '2h', '90m', '30s', a
    bare number (seconds), or combos like '1h30m'."""
    s = s.strip().lower()
    units = {"h": 3600, "m": 60, "s": 1}
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", s)
    if matches:
        return sum(float(val) * units[unit] for val, unit in matches)
    return float(s)  # bare seconds


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
    parser.add_argument(
        "--max-time",
        default=None,
        help="Wall-clock search limit, e.g. '2h', '90m', '3600s', '1h30m', or "
        "bare seconds. The search stops on whichever comes first: this or "
        "--max-fe. Omit for FE-only (default).",
    )
    # --- Ablation switches (omit for full qpilot) ---
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Ablation: disable the HD-KG (no traits/hypotheses/confidence updates).",
    )
    parser.add_argument(
        "--no-crossover",
        action="store_true",
        help="Ablation: disable the crossover operator (mutation-only evolution).",
    )
    parser.add_argument(
        "--no-mutation",
        action="store_true",
        help="Ablation: disable the mutation operator (crossover-only evolution).",
    )
    parser.add_argument(
        "--no-reideation",
        action="store_true",
        help="Ablation: disable re-ideation; keep running crossover+mutation "
        "until the FE budget is exhausted.",
    )
    parser.add_argument(
        "--no-ideation-memory",
        action="store_true",
        help="Ablation: first idea generation ignores past-experiment memory "
        "(no resume injected). Does not affect evolution seeding or the KG.",
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
        run_evolution=True,
        experiment_name=args.experiment_name,
        seed=args.seed,
        max_time_seconds=_parse_duration(args.max_time) if args.max_time else None,
        use_kg=not args.no_kg,
        use_crossover=not args.no_crossover,
        use_mutation=not args.no_mutation,
        use_reideation=not args.no_reideation,
        use_ideation_memory=not args.no_ideation_memory,
    )

    qpilot = Qpilot(config)
    qpilot.run()


if __name__ == "__main__":
    main()
