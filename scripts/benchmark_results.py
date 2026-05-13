import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import *
from src.graph.graph import *
from qpu.src.load_backend import *
import numpy as np
import time
from tqdm import tqdm 

BACKEND = "ibm_sherbrooke"
BENCHMARK_DIR = "/home/jmsb00nd/Documents/auto-heuristics/benchmarks/qasmbench-large"

edges = load_backend_edges(BACKEND)
circuit_files = list(Path(BENCHMARK_DIR).glob("*.json"))

if not circuit_files:
    print(f"Error: No .json files found in {BENCHMARK_DIR}")
    exit()

print(f"Starting batch run on {len(circuit_files)} circuits...")

# Initialize accumulators for calculating the average
total_swaps = 0
total_depth = 0
successful_circuits = 0

pbar = tqdm(circuit_files, desc="Processing Circuits", unit="circuit")

for circuit_path in pbar:
    try:
        pbar.set_postfix(file=circuit_path.name[:15]) 
        
        data = json_file_to_isl(str(circuit_path))
        router = Qlosure(edges, data)
        
        start_time = time.time()
                        
        min_swaps, min_depth, _ = router.run()
                
        end_time = time.time()
        runtime = end_time - start_time
        
        # Add to totals instead of writing to file
        total_swaps += min_swaps
        total_depth += min_depth
        successful_circuits += 1

    except Exception as e:
        # tqdm.write ensures the error doesn't break the progress bar formatting
        tqdm.write(f"Failed to process {circuit_path.name}: {e}")

print("\nBatch run completed.")

# Calculate and print the averages
if successful_circuits > 0:
    avg_swaps = total_swaps / successful_circuits
    avg_depth = total_depth / successful_circuits
    
    print("-" * 30)
    print("RESULTS SUMMARY")
    print("-" * 30)
    print(f"Circuits Processed : {successful_circuits}/{len(circuit_files)}")
    print(f"Average Swaps      : {avg_swaps:.2f}")
    print(f"Average Depth      : {avg_depth:.2f}")
    print("-" * 30)
else:
    print("No circuits were successfully processed.")