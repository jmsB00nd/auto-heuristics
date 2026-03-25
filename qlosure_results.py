from pathlib import Path
from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import *
from src.graph.graph import *
from qpu.src.load_backend import *
import numpy as np
import time
import csv
import os
from tqdm import tqdm 

BACKEND = "ibm_sherbrooke"
BENCHMARK_DIR = "/home/jmsb00nd/Documents/auto-heuristics/benchmarks/qasmbench-large"
OUTPUT_DIR = "/home/jmsb00nd/Documents/auto-heuristics/autoheuristics_results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "init_mapping_llm_qasm_bench_ibm_sherbrooke_trivial.csv")

edges = load_backend_edges(BACKEND)
circuit_files = list(Path(BENCHMARK_DIR).glob("*.json"))

if not circuit_files:
    print(f"Error: No .json files found in {BENCHMARK_DIR}")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting batch run on {len(circuit_files)} circuits...")
print(f"Saving results to: {OUTPUT_CSV}")

with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "final_depth", "swap_count", "runtime"])

    pbar = tqdm(circuit_files, desc="Processing Circuits", unit="circuit")

    for circuit_path in pbar:
        try:
            pbar.set_postfix(file=circuit_path.name[:15]) 
            
            data = json_file_to_isl(str(circuit_path))
            router = Qlosure(edges, data)
            
            start_time = time.time()
                            
            min_swaps, min_depth, _ = router.run(heuristic_method="Qlosure", initial_mapping_method="trivial")
                    
            end_time = time.time()
            runtime = end_time - start_time
            
            writer.writerow([circuit_path.name, min_depth, min_swaps, runtime])

        except Exception as e:
            # tqdm.write ensures the error doesn't break the progress bar formatting
            tqdm.write(f"Failed to process {circuit_path.name}: {e}")

print("Batch run completed.")