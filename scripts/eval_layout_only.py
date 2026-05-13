"""Apples-to-apples evaluation of *initial mapping* quality only.

Both TIG and SABRE provide the initial layout; the SAME router (Qlosure with
num_iter=1, no bi-directional pass) is run downstream. This isolates the
contribution of the initial mapping heuristic.
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


def eval_one(circuit_path, backend):
    data = json_file_to_isl(circuit_path)
    edges = load_backend_edges(backend)

    q1 = Qlosure(edges, data, with_circuit=False)
    t0 = time.time()
    sa_sw, sa_d, _ = q1.run(initial_mapping_method="sabre",
                            heuristic_method="qlosure",
                            num_iter=1, verbose=0)
    sa_t = time.time() - t0

    q2 = Qlosure(edges, data, with_circuit=False)
    t0 = time.time()
    ti_sw, ti_d, _ = q2.run(initial_mapping_method="tig",
                            heuristic_method="qlosure",
                            num_iter=1, verbose=0)
    ti_t = time.time() - t0

    return {
        "circuit": os.path.basename(circuit_path),
        "sabre_swaps": sa_sw, "sabre_depth": sa_d, "sabre_time": sa_t,
        "tig_swaps": ti_sw, "tig_depth": ti_d, "tig_time": ti_t,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="benchmarks/queko-bss-16qbt")
    parser.add_argument("--backend", type=str, default="ankaa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    bench_dir = Path(args.bench)
    files = sorted(bench_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]

    results = []
    print(f"{'circuit':40s} {'sabre_sw':>8s} {'tig_sw':>8s} {'d_sw%':>7s} | "
          f"{'sabre_d':>8s} {'tig_d':>8s} {'d_d%':>7s}", flush=True)
    for f in files:
        try:
            r = eval_one(str(f), args.backend)
        except Exception as e:
            print(f"{f.name:40s} ERROR {e}", flush=True)
            continue
        s = r["sabre_swaps"]
        t = r["tig_swaps"]
        if s == 0:
            dsw = 0.0 if t == 0 else float('-inf')
        else:
            dsw = 100.0 * (s - t) / s
        dd = 100.0 * (r["sabre_depth"] - r["tig_depth"]) / r["sabre_depth"] if r["sabre_depth"] else 0
        print(f"{r['circuit']:40s} {s:8d} {t:8d} {dsw:>+6.1f}% | "
              f"{r['sabre_depth']:8d} {r['tig_depth']:8d} {dd:>+6.1f}%", flush=True)
        results.append(r)

    tot_s = sum(r["sabre_swaps"] for r in results)
    tot_t = sum(r["tig_swaps"] for r in results)
    tot_sd = sum(r["sabre_depth"] for r in results)
    tot_td = sum(r["tig_depth"] for r in results)
    n = len(results)
    valid_sw = [r for r in results if r["sabre_swaps"]]
    avg_dsw = (sum(100.0 * (r["sabre_swaps"] - r["tig_swaps"]) / r["sabre_swaps"]
                    for r in valid_sw) / max(1, len(valid_sw))) if valid_sw else 0.0
    avg_dd = sum((100.0 * (r["sabre_depth"] - r["tig_depth"]) / r["sabre_depth"])
                 for r in results if r["sabre_depth"]) / max(1, n)

    print("=" * 100, flush=True)
    print(f"N={n}  total SABRE-init swaps={tot_s}  TIG-init swaps={tot_t}  "
          f"(global d={100.0 * (tot_s - tot_t)/max(1,tot_s):+.2f}%, "
          f"avg per-circuit d={avg_dsw:+.2f}%)", flush=True)
    print(f"         total SABRE-init depth={tot_sd}  TIG-init depth={tot_td}  "
          f"(global d={100.0 * (tot_sd - tot_td)/max(1,tot_sd):+.2f}%, "
          f"avg per-circuit d={avg_dd:+.2f}%)", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved JSON to {args.out}", flush=True)


if __name__ == "__main__":
    main()
