import threading
import time
import types
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from config import OrchestratorConfig
from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import json_file_to_isl
from qpu.src.load_backend import load_backend_edges

class CodeEvaluator:
    """Executes untrusted heuristic code and evaluates performance."""
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.edges = load_backend_edges(self.config.backend)
        self.circuit_files = list(Path(self.config.benchmark_dir).glob("*.json"))

    def evaluate(self, heuristic_code_str: str) -> Dict[str, Any]:
        local_scope = {}
        try:
            exec(heuristic_code_str, globals(), local_scope)
        except Exception as e:
            return {"mean_swaps": float('inf'), "error": f"Syntax Error: {str(e)}"}

        heuristic_func = self._extract_function(local_scope)
        if not heuristic_func:
            return {"mean_swaps": float('inf'), "error": "Function 'qlosure_poly_heuristic' not found."}

        if not self.circuit_files:
            return {"error": f"No .json files found in {self.config.benchmark_dir}"}

        return self._run_benchmarks(heuristic_func)

    def _extract_function(self, local_scope: dict) -> Optional[callable]:
        func = local_scope.get('qlosure_poly_heuristic')
        if not func:
            for obj in local_scope.values():
                if isinstance(obj, type) and hasattr(obj, 'qlosure_poly_heuristic'):
                    return getattr(obj, 'qlosure_poly_heuristic')
        return func

    def _run_benchmarks(self, heuristic_func: callable) -> Dict[str, Any]:
        results = {"swaps": [], "depth": [], "runtimes": [], "failures": []}

        for circuit_path in tqdm(self.circuit_files, desc="Routing progress"):
            data = json_file_to_isl(str(circuit_path))
            router = Qlosure(self.edges, data)
            router.qlosure_poly_heuristic = types.MethodType(heuristic_func, router)

            res_container, exc_container = {}, {}

            def _target():
                try:
                    swaps, depth, _ = router.run(heuristic_method="Qlosure")
                    res_container['swaps'], res_container['depth'] = swaps, depth
                except Exception as e:
                    exc_container['error'] = e

            start_time = time.time()
            thread = threading.Thread(target=_target, daemon=True)
            thread.start()
            thread.join(timeout=self.config.timeout_seconds)
            
            if thread.is_alive():
                return {"mean_swaps": float('inf'), "error": f"Timeout on {circuit_path.name}"}

            if 'error' in exc_container:
                results["failures"].append({"circuit": circuit_path.name, "error": str(exc_container['error'])})
                # Abort early if the heuristic is clearly broken
                if len(results["failures"]) >= 3:
                    break 
            else:
                results["swaps"].append(res_container['swaps'])
                results["depth"].append(res_container['depth'])
                results["runtimes"].append(time.time() - start_time)

        if len(results["failures"]) > 0:
            error_msg = f"Failed on {len(results['failures'])} circuits. First error: {results['failures'][0]['error']}"
            if len(results["failures"]) >= 3:
                error_msg = "Aborted early: " + error_msg
                
            return {
                "mean_swaps": float('inf'),
                "mean_depth": 0,
                "error": error_msg,
                "failures": results["failures"]
            }

        # If it reaches here, it truly succeeded on all circuits
        return {
            "mean_swaps": np.mean(results["swaps"]) if results["swaps"] else float('inf'),
            "mean_depth": np.mean(results["depth"]) if results["depth"] else 0,
            "error": None if results["swaps"] else "All circuits failed / No circuits found",
            "failures": []
        }