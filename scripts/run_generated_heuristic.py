"""Run a generated `init_mapping` heuristic over a benchmark and write a CSV.

The heuristic file is expected to define a top-level `init_mapping(self)`
function (the same shape produced under `outputs/logs/<run>/heuristics/`).
Output schema matches the canonical generated-heuristic CSVs in
`experiments_results/<benchmark>/`:

    filename,final_depth,swap_count,runtime

Usage:
    python scripts/run_generated_heuristic.py \\
        --heuristic outputs/logs/<run>/heuristics/<name>.py \\
        --benchmark queko-bss-16qbt
"""

import argparse
import csv
import inspect
import signal
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import json_file_to_isl
from qpu.src.load_backend import load_backend_edges


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_FUNC = "init_mapping"


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def load_heuristic(heuristic_path: Path):
    code = heuristic_path.read_text()
    local_scope: dict = {}
    exec(compile(code, str(heuristic_path), "exec"), {}, local_scope)

    func = local_scope.get(TARGET_FUNC)
    if func is None:
        for obj in local_scope.values():
            if isinstance(obj, type) and TARGET_FUNC in obj.__dict__:
                raw = inspect.getattr_static(obj, TARGET_FUNC)
                func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
                break
    if func is None:
        raise RuntimeError(f"`{TARGET_FUNC}` not found in {heuristic_path}")
    return func


def run_circuit(circuit_path: Path, edges, init_func, time_budget: int):
    data = json_file_to_isl(str(circuit_path))
    router = Qlosure(edges, data)
    setattr(router, TARGET_FUNC, types.MethodType(init_func, router))

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(time_budget)
    start = time.time()
    try:
        swaps, depth, _ = router.run()
    finally:
        signal.alarm(0)
    return swaps, depth, time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--heuristic", required=True, type=Path,
                        help="Path to a generated heuristic .py file (defining init_mapping)")
    parser.add_argument("-b", "--benchmark", required=True,
                        help="Benchmark folder name under benchmarks/ (e.g. queko-bss-16qbt)")
    parser.add_argument("--backend", default="ibm_sherbrooke",
                        help="Backend topology to load")
    parser.add_argument("--output-name", default=None,
                        help="Output CSV name (without .csv). Defaults to heuristic file stem.")
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Per-circuit timeout in seconds")
    args = parser.parse_args()

    heuristic_path: Path = args.heuristic.resolve()
    if not heuristic_path.is_file():
        sys.exit(f"heuristic file not found: {heuristic_path}")

    benchmark_dir = REPO_ROOT / "benchmarks" / args.benchmark
    if not benchmark_dir.is_dir():
        sys.exit(f"benchmark folder not found: {benchmark_dir}")

    circuit_files = sorted(benchmark_dir.glob("*.json"))
    if not circuit_files:
        sys.exit(f"no .json circuits in {benchmark_dir}")

    out_dir = REPO_ROOT / "experiments_results" / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name or heuristic_path.stem
    out_csv = out_dir / f"{out_name}.csv"

    print(f"heuristic : {heuristic_path}")
    print(f"benchmark : {benchmark_dir}  ({len(circuit_files)} circuits)")
    print(f"backend   : {args.backend}")
    print(f"output    : {out_csv}")

    init_func = load_heuristic(heuristic_path)
    edges = load_backend_edges(args.backend)

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "final_depth", "swap_count", "runtime"])

        for circuit_path in tqdm(circuit_files, desc="circuits", unit="circuit"):
            try:
                swaps, depth, runtime = run_circuit(circuit_path, edges, init_func, args.time_budget)
                writer.writerow([circuit_path.name, depth, swaps, runtime])
                f.flush()
            except _Timeout:
                tqdm.write(f"TIMEOUT  {circuit_path.name}")
                writer.writerow([circuit_path.name, "timeout", "timeout", args.time_budget])
                f.flush()
            except Exception as e:
                tqdm.write(f"FAIL     {circuit_path.name}: {e}")
                writer.writerow([circuit_path.name, "error", "error", "error"])
                f.flush()

    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
