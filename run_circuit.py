import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Any

from qiskit.qasm2 import dump

from src.utils.isl_data_loader import json_file_to_isl
from src.mapping.routing import Qlosure
from qpu.src.load_backend import load_backend_edges


# Argument parser setup
parser = argparse.ArgumentParser(
    description="Run Qlosure with optional parameters")
parser.add_argument("--circuit", type=str,
                    default="/home/jmsb00nd/Documents/auto-heuristics/benchmarks/qasmbench-large/multiplier_n75__1308CYC.json", help="Path to circuit JSON file")
parser.add_argument("--backend", type=str,
                    default="ibm_sherbrooke", help="Name of the backend")
parser.add_argument("--initial", type=str, default="trivial",
                    help="Initial mapping method")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
parser.add_argument("--heuristic", type=str, default="qlosure",
                    help="Heuristic to use for mapping")
parser.add_argument("--num_iterations", type=int, default=1,
                    help="number of bidirectional passes")
parser.add_argument("--competitors", action="store_true",
                    help="Run and compare with competitor mappers")

args = parser.parse_args()

# Load circuit data
print(f"Loading circuit from: {args.circuit}")
data = json_file_to_isl(args.circuit)
print("✅ Circuit loaded successfully.")

# Load backend edges
print(f"Loading backend: {args.backend}")
edges = load_backend_edges(args.backend)
print("✅ Backend topology loaded.")

# Run Qlosure
poly_mapper = Qlosure(edges, data)
qlosure_results = poly_mapper.run(initial_mapping_method=args.initial, verbose=args.verbose,
                                  heuristic_method=args.heuristic, num_iter=args.num_iterations)
# Store results
results = {
    "qlosure": {"swaps": qlosure_results[0], "depth": qlosure_results[1], "time": qlosure_results[2]},
}

base_name = os.path.splitext(os.path.basename(args.circuit))[0]
folder = os.path.join("tmp_results", base_name)
os.makedirs(folder, exist_ok=True)

# --- Save mapped QASM ---
timestamp = time.strftime("%Y%m%d-%H%M%S")
qasm_filename = f"compiled_circuit_{timestamp}.qasm"
qasm_path = os.path.join(folder, qasm_filename)

with open(qasm_path, "w") as f:
    dump(poly_mapper.circuit, f)

# --- Save stats JSON ---
results = {
    "qlosure": {
        "swaps": qlosure_results[0],
        "depth": qlosure_results[1],
        "time": qlosure_results[2]
    }
}

stats_path = os.path.join(folder, f"stats_{timestamp}.json")
with open(stats_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ QASM file saved to: {qasm_path}")

# Print results in table format
print("\n📊 Mapping Results")
print("+-----------+--------+--------+")
print("| Method    | Swaps  | Depth  |")
print("+-----------+--------+--------+")
for method, res in results.items():
    print(f"| {method:<9} | {res['swaps']:<6} | {res['depth']:<6} |")
print("+-----------+--------+--------+")
