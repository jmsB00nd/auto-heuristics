import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

TIMEOUT_SECONDS = 30


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.isl_data_loader import json_file_to_isl
from src.mapping.routing import Qlosure
from qpu.src.load_backend import load_backend_edges


parser = argparse.ArgumentParser(
    description="Run Qlosure on every circuit in a benchmark folder")
parser.add_argument("--benchmark", type=str,
                    default="/home/jmsb00nd/Documents/auto-heuristics/benchmarks/qasmbench-medium",
                    help="Path to a benchmark folder containing circuit JSON files")
parser.add_argument("--backend", type=str,
                    default="ibm_sherbrooke", help="Name of sabre backend")
parser.add_argument("--initial", type=str, default="qmap",
                    help="Initial mapping method: 'cblc' , 'sabre', 'trivial', 'random'.")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--heuristic", type=str, default="qlosure",
                    help="Heuristic to use for routing")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional routing passes")
parser.add_argument("--competitors", action="store_true",
                    help="Run and compare with competitor mappers")
parser.add_argument("--output_dir", type=str,
                    default="/home/jmsb00nd/Documents/auto-heuristics/experiments_results/tmp/tmp",
                    help="Directory where the result CSV is written")

args = parser.parse_args()

bench_dir = Path(args.benchmark)
if not bench_dir.is_dir():
    sys.exit(f"❌ Benchmark folder not found: {bench_dir}")

circuits = sorted(bench_dir.glob("*.json"))
if not circuits:
    sys.exit(f"❌ No .json circuits found in {bench_dir}")

edges = load_backend_edges(args.backend)
print(f"✅ Backend '{args.backend}' loaded — {len(circuits)} circuits queued.")

if args.competitors:
    from baselines.pytket import run_pytket
    from baselines.sabre import run_sabre
    from baselines.qmap import run_qmap
    from baselines.cirq import run_cirq

fieldnames = ["filename", "depth", "qops", "swap_count", "final_depth", "runtime", "init_time"]
if args.competitors:
    fieldnames += [
        "sabre_swaps", "sabre_depth",
        "qmap_swaps", "qmap_depth",
        "tket_swaps", "tket_depth",
        "cirq_swaps", "cirq_depth",
    ]

rows = []
tot_sw = tot_dp = 0.0
ok = 0
failures = []

print(f"\n{'circuit':<40}{'swaps':>8}{'depth':>8}{'sec':>9}")
print("-" * 65)

for circ_path in circuits:
    try:
        data = json_file_to_isl(str(circ_path))
    except Exception as e:
        print(f"{circ_path.name:<40}LOAD FAIL: {e}")
        failures.append(circ_path.name)
        continue

    qops = data.get("Stats", {}).get("Qops") if isinstance(data.get("Stats"), dict) else None

    original_depth = None
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        try:
            mapper = Qlosure(edges, data)
            original_depth = mapper.original_circuit.depth() if mapper.with_circuit else None
            sw, dp, sec = mapper.run(
                initial_mapping_method=args.initial,
                verbose=args.verbose,
                heuristic_method=args.heuristic,
                num_iter=args.num_iterations,
            )
        finally:
            signal.alarm(0)
    except TimeoutException:
        row = {
            "filename": str(circ_path),
            "depth": original_depth,
            "qops": qops,
            "swap_count": "timeout",
            "final_depth": "timeout",
            "runtime": TIMEOUT_SECONDS,
            "init_time": TIMEOUT_SECONDS,
        }
        rows.append(row)
        print(f"{circ_path.name:<40}{'timeout':>8}{'timeout':>8}{float(TIMEOUT_SECONDS):>9.2f}")
        continue
    except Exception as e:
        print(f"{circ_path.name:<40}RUN FAIL: {e}")
        failures.append(circ_path.name)
        continue

    row = {
        "filename": str(circ_path),
        "depth": original_depth,
        "qops": qops,
        "swap_count": sw,
        "final_depth": dp,
        "runtime": round(sec, 4),
        "init_time": round(mapper.init_time, 4),
    }

    if args.competitors:
        try:
            cirq_res = run_cirq(data, edges, initial_mapping=args.initial)
            sabre_res = run_sabre(data, edges, layout=args.initial)
            qmap_res = run_qmap(data, edges, initial_mapping=args.initial)
            tket_res = run_pytket(data, edges, initial_mapping=args.initial)
            row.update({
                "sabre_swaps": sabre_res["swaps"], "sabre_depth": sabre_res["depth"],
                "qmap_swaps": qmap_res["swaps"], "qmap_depth": qmap_res["depth"],
                "tket_swaps": tket_res["swaps"], "tket_depth": tket_res["depth"],
                "cirq_swaps": cirq_res["swaps"], "cirq_depth": cirq_res["depth"],
            })
        except Exception as e:
            print(f"  ⚠ competitors failed on {circ_path.name}: {e}")

    rows.append(row)
    tot_sw += sw
    tot_dp += dp
    ok += 1
    print(f"{circ_path.name:<40}{sw:>8}{dp:>8}{sec:>9.2f}")

print("-" * 65)
if ok:
    mean_sw = tot_sw / ok
    mean_dp = tot_dp / ok
    print(f"\n📊 Mean over {ok} circuits — swaps: {mean_sw:.2f}  depth: {mean_dp:.2f}")
else:
    print("No circuits completed successfully.")
if failures:
    print(f"⚠ failures ({len(failures)}): {failures}")

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{bench_dir.name}_{args.heuristic}_{args.initial}.csv"

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Results saved to: {out_path}")
