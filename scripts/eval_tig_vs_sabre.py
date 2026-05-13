"""Evaluate TIG initial mapping vs SABRE baseline (Qiskit) on a benchmark set.

Single-pass: TIG uses no routing pre-run. Compared against `run_sabre` which
uses Qiskit's SabreLayout + SabreSwap. Reports per-circuit swaps, depth and
aggregate deltas.
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.isl_data_loader import json_file_to_isl
from src.mapping.routing import Qlosure
from qpu.src.load_backend import load_backend_edges
from baselines.sabre import run_sabre


def eval_one(circuit_path, backend, num_iter=1):
    data = json_file_to_isl(circuit_path)
    edges = load_backend_edges(backend)

    t0 = time.time()
    sabre = run_sabre(data, edges, layout="sabre")
    sabre_time = time.time() - t0

    q = Qlosure(edges, data, with_circuit=False)
    t0 = time.time()
    swaps, depth, _ = q.run(initial_mapping_method="tig",
                            heuristic_method="qlosure",
                            num_iter=num_iter, verbose=0)
    tig_time = time.time() - t0

    return {
        "circuit": os.path.basename(circuit_path),
        "sabre_swaps": sabre["swaps"], "sabre_depth": sabre["depth"], "sabre_time": sabre_time,
        "tig_swaps": swaps, "tig_depth": depth, "tig_time": tig_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="benchmarks/queko-bss-16qbt")
    parser.add_argument("--backend", type=str, default="ankaa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    bench_dir = Path(args.bench)
    files = sorted(bench_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]

    results = []
    print(f"{'circuit':40s} {'sabre_sw':>8s} {'tig_sw':>8s} {'d_sw%':>7s} | "
          f"{'sabre_d':>8s} {'tig_d':>8s} {'d_d%':>7s}")
    for f in files:
        try:
            r = eval_one(str(f), args.backend, num_iter=args.num_iter)
        except Exception as e:
            print(f"{f.name:40s} ERROR {e}")
            continue
        s = r["sabre_swaps"]
        t = r["tig_swaps"]
        if s == 0:
            dsw = 0.0 if t == 0 else float('inf')
        else:
            dsw = 100.0 * (s - t) / s
        dd = 100.0 * (r["sabre_depth"] - r["tig_depth"]) / r["sabre_depth"] if r["sabre_depth"] else 0
        print(f"{r['circuit']:40s} {s:8d} {t:8d} {dsw:>+6.1f}% | "
              f"{r['sabre_depth']:8d} {r['tig_depth']:8d} {dd:>+6.1f}%")
        results.append(r)

    tot_s = sum(r["sabre_swaps"] for r in results)
    tot_t = sum(r["tig_swaps"] for r in results)
    tot_sd = sum(r["sabre_depth"] for r in results)
    tot_td = sum(r["tig_depth"] for r in results)
    n = len(results)
    avg_dsw = sum((100.0 * (r["sabre_swaps"] - r["tig_swaps"]) / r["sabre_swaps"])
                  for r in results if r["sabre_swaps"]) / max(1, sum(1 for r in results if r["sabre_swaps"]))
    avg_dd = sum((100.0 * (r["sabre_depth"] - r["tig_depth"]) / r["sabre_depth"])
                 for r in results if r["sabre_depth"]) / max(1, n)

    print("=" * 100)
    print(f"N={n}  total SABRE swaps={tot_s}  TIG swaps={tot_t}  "
          f"(global d={100.0 * (tot_s - tot_t)/max(1,tot_s):+.2f}%, "
          f"avg per-circuit d={avg_dsw:+.2f}%)")
    print(f"         total SABRE depth={tot_sd}  TIG depth={tot_td}  "
          f"(global d={100.0 * (tot_sd - tot_td)/max(1,tot_sd):+.2f}%, "
          f"avg per-circuit d={avg_dd:+.2f}%)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved JSON to {args.out}")


if __name__ == "__main__":
    main()
