import inspect
import multiprocessing as mp
import time
import types
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from .config import OrchestratorConfig
from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import json_file_to_isl
from qpu.src.load_backend import load_backend_edges


def _worker(heuristic_code_str: str, target_func_name: str, edges, data, queue: "mp.Queue"):
    try:
        local_scope: dict = {}
        exec(heuristic_code_str, globals(), local_scope)

        func = local_scope.get(target_func_name)
        if func is None:
            for obj in local_scope.values():
                if isinstance(obj, type) and target_func_name in obj.__dict__:
                    raw = inspect.getattr_static(obj, target_func_name)
                    if isinstance(raw, staticmethod):
                        func = raw.__func__
                    elif isinstance(raw, classmethod):
                        func = raw.__func__.__get__(obj, type(obj))
                    else:
                        func = raw
                    break

        if func is None:
            queue.put(("error", f"Function '{target_func_name}' not found."))
            return

        router = Qlosure(edges, data)
        setattr(router, target_func_name, types.MethodType(func, router))
        swaps, depth, _ = router.run()
        queue.put(("ok", (swaps, depth)))
    except Exception as e:
        queue.put(("error", str(e)))


class CodeEvaluator:
    """Executes untrusted heuristic code and evaluates performance."""
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.edges = load_backend_edges(self.config.backend)
        self.circuit_files = list(Path(self.config.benchmark_dir).glob("*.json"))
        self._mp_ctx = mp.get_context("fork")

    @property
    def target_func_name(self) -> str:
        return "init_mapping" if self.config.problem == "mapping" else "qlosure_poly_heuristic"

    def evaluate(self, heuristic_code_str: str) -> Dict[str, Any]:
        try:
            compile(heuristic_code_str, "<heuristic>", "exec")
        except SyntaxError as e:
            return {"mean_swaps": float('inf'), "error": f"Syntax Error: {str(e)}"}

        if not self.circuit_files:
            return {"error": f"No .json files found in {self.config.benchmark_dir}"}

        return self._run_benchmarks(heuristic_code_str)

    def _run_benchmarks(self, heuristic_code_str: str) -> Dict[str, Any]:
        results: Dict[str, list] = {"swaps": [], "depth": [], "runtimes": [], "failures": []}
        target = self.target_func_name
        timeout = self.config.timeout_seconds

        for circuit_path in tqdm(self.circuit_files, desc="Run progress"):
            data = json_file_to_isl(str(circuit_path))
            queue: "mp.Queue" = self._mp_ctx.Queue()
            proc = self._mp_ctx.Process(
                target=_worker,
                args=(heuristic_code_str, target, self.edges, data, queue),
                daemon=True,
            )

            start_time = time.time()
            proc.start()
            proc.join(timeout=timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                results["failures"].append({"circuit": circuit_path.name, "error": "Timeout"})
                if len(results["failures"]) >= 3:
                    break
                continue

            try:
                status, payload = queue.get_nowait()
            except Exception:
                status, payload = "error", f"Worker exited without result (exitcode={proc.exitcode})"

            if status == "ok":
                swaps, depth = payload
                results["swaps"].append(swaps)
                results["depth"].append(depth)
                results["runtimes"].append(time.time() - start_time)
            else:
                results["failures"].append({"circuit": circuit_path.name, "error": str(payload)})
                if len(results["failures"]) >= 3:
                    break

        if results["failures"]:
            error_msg = f"Failed on {len(results['failures'])} circuits. First error: {results['failures'][0]['error']}"
            if len(results["failures"]) >= 3:
                error_msg = "Aborted early: " + error_msg
            return {
                "mean_swaps": float('inf'),
                "mean_depth": 0,
                "error": error_msg,
                "failures": results["failures"],
            }

        return {
            "mean_swaps": float(np.mean(results["swaps"])) if results["swaps"] else float('inf'),
            "mean_depth": float(np.mean(results["depth"])) if results["depth"] else 0,
            "error": None if results["swaps"] else "All circuits failed / No circuits found",
            "failures": [],
        }
