"""Per-circuit benchmark with hard 5-min/circuit timeout.

Usage:
    python3 scripts/benchmark_detailed.py            # all 22 circuits
    python3 scripts/benchmark_detailed.py fast       # subset of 6 fast ones
    python3 scripts/benchmark_detailed.py adder      # any substring filter
"""

import sys
import signal
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import json_file_to_isl
from qpu.src.load_backend import load_backend_edges

BACKEND = "ibm_sherbrooke"
BENCHMARK_DIR = "/home/jmsb00nd/Documents/auto-heuristics/benchmarks/qasmbench-large"

# Sabre baseline numbers for relative deltas (router avg 501.91 sw / 910.64 dp).
BASELINE_SABRE = {
    "adder_n28__42CYC": (77, 293),
    "adder_n64__78CYC": (246, 571),
    "bv_n30__22CYC": (16, 55),
    "bv_n70__40CYC": (42, 128),
    "cat_n35__36CYC": (7, 44),
    "cat_n65__66CYC": (14, 81),
    "dnn_n51__58CYC": (154, 442),
    "ghz_n40__41CYC": (8, 50),
    "ghz_n78__79CYC": (16, 96),
    "ising_n34__16CYC": (7, 19),
    "ising_n66__16CYC": (13, 19),
    "knn_n31__18CYC": (34, 207),
    "knn_n67__36CYC": (141, 463),
    "multiplier_n45__462CYC": (1499, 3407),
    "multiplier_n75__1308CYC": (5628, 10198),
    "qft_n29__222CYC": (427, 498),
    "qft_n63__494CYC": (2259, 1556),
    "qugan_n39__40CYC": (106, 347),
    "qugan_n71__72CYC": (266, 803),
    "swap_test_n41__23CYC": (57, 286),
    "wstate_n36__73CYC": (8, 151),
    "wstate_n76__153CYC": (17, 320),
}

# Subset that captures most variance fast (~15s total).
FAST_SET = [
    "qft_n29__222CYC",
    "multiplier_n45__462CYC",
    "qft_n63__494CYC",
    "adder_n64__78CYC",
    "knn_n67__36CYC",
    "qugan_n71__72CYC",
]


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def run_one(path, time_budget=300):
    data = json_file_to_isl(str(path))
    edges = load_backend_edges(BACKEND)
    router = Qlosure(edges, data)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(time_budget)
    t0 = time.time()
    try:
        sw, dp, _ = router.run()
    finally:
        signal.alarm(0)
    return sw, dp, time.time() - t0


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(Path(BENCHMARK_DIR).glob("*.json"))

    if arg == "fast":
        files = [p for p in files if p.stem in FAST_SET]
    elif arg:
        files = [p for p in files if arg in p.name]

    n = len(files)
    print(f"Running {n} circuits with 5-min/circuit cap.")
    print(f"{'circuit':<28}{'swaps':>8}{'depth':>8}{'Δsw%':>9}{'Δdp%':>9}{'sec':>8}")
    print("-" * 70)

    tot_sw = tot_dp = base_sw = base_dp = 0
    ok = 0
    fail = []
    for p in files:
        stem = p.stem
        try:
            sw, dp, sec = run_one(p)
        except _Timeout:
            print(f"{stem:<28}{'TIMEOUT':>8}")
            fail.append(stem)
            continue
        except Exception as e:
            print(f"{stem:<28}FAIL: {e}")
            fail.append(stem)
            continue

        b_sw, b_dp = BASELINE_SABRE.get(stem, (sw, dp))
        d_sw = (sw - b_sw) / b_sw * 100 if b_sw else 0.0
        d_dp = (dp - b_dp) / b_dp * 100 if b_dp else 0.0
        print(f"{stem:<28}{sw:>8}{dp:>8}{d_sw:>+8.1f}%{d_dp:>+8.1f}%{sec:>8.1f}")
        tot_sw += sw
        tot_dp += dp
        base_sw += b_sw
        base_dp += b_dp
        ok += 1

    print("-" * 70)
    if ok:
        print(f"avg swaps : {tot_sw/ok:.2f}   (sabre {base_sw/ok:.2f})  Δ {(tot_sw-base_sw)/base_sw*100:+.2f}%")
        print(f"avg depth : {tot_dp/ok:.2f}   (sabre {base_dp/ok:.2f})  Δ {(tot_dp-base_dp)/base_dp*100:+.2f}%")
    if fail:
        print(f"failures: {fail}")


if __name__ == "__main__":
    main()
