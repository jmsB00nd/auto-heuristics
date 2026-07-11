# Module Name: QuantumMappingEvaluation
# Last Revision: 2026/3/2
# Description: Evaluates a heuristic cost function for quantum circuit routing.
#              Given a quantum circuit and a hardware backend coupling map,
#              the goal is to minimise the number of SWAP gates inserted so that
#              every 2-qubit gate acts on physically adjacent qubits.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) per circuit: int (default: 60).
#    - n_instance: Number of circuit instances to evaluate: int (default: 16).
#    - backend: Hardware backend name (must be registered in qpu/src/load_backend.py): str (default: "fake_20q_v1").
#    - benchmark_dir: Path to directory containing circuit JSON files: str (default: benchmarks/queko-bss-16qbt).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# Ensure the quantum_initial_mapping root is on sys.path so src.* and qpu.* imports resolve
_QM_DIR = os.path.dirname(os.path.abspath(__file__))
if _QM_DIR not in sys.path:
    sys.path.insert(0, _QM_DIR)

from src.mapping.routing import Qlosure
from src.utils.isl_data_loader import json_file_to_isl
from qpu.src.load_backend import load_backend_edges

from llm4ad.base import Evaluation
from llm4ad.task.optimization.quantum_initial_mapping.template import template_program, task_description

__all__ = ['QuantumInitialMappingEvaluation']


class QuantumInitialMappingEvaluation(Evaluation):
    """Evaluates a heuristic cost function for quantum circuit initial mapping."""

    def __init__(self,
                 timeout_seconds: int = 300,
                 n_instance: int = 22,
                 backend: str = "ibm_sherbrooke",
                 benchmark_dir: str = None,
                 **kwargs):
        """
        Args:
            timeout_seconds: Maximum seconds allowed per circuit evaluation.
            n_instance: Number of circuit benchmark files to use. None means use all files in the directory.
            backend: Hardware backend name registered in qpu/src/load_backend.py.
            benchmark_dir: Path to benchmark JSON directory.
                           Defaults to benchmarks/qasmbench-large relative to this file.
        """
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self.n_instance = n_instance
        self.backend = backend
        self.benchmark_dir = benchmark_dir if benchmark_dir is not None else \
            os.path.join(_QM_DIR, "benchmarks", "qasmbench-large")

        self._edges = load_backend_edges(self.backend)
        circuit_files = sorted(Path(self.benchmark_dir).glob("*.json"))
        if self.n_instance is not None:
            circuit_files = circuit_files[:self.n_instance]
        self._datasets = [
            (circuit_path, json_file_to_isl(str(circuit_path)))
            for circuit_path in circuit_files
        ]

    def evaluate_program(self, program_str: str, callable_func: callable = None) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, eva: callable) -> float | None:
        """
        Route each benchmark circuit using the supplied initial mapping heuristic.
        
        Args:
            eva: A function with signature:
                 def generate_New_method_initial_mapping(
                     distance_matrix,
                     num_qubits,
                     backend,
                     backend_connections,
                     access,
                     dag_dependencies_count,
                     dag2q,
                     dag_predecessors2q,
                 ) -> Tuple[List[int], List[int]]
                 Returns (mapping, reverse_mapping) where:
                   - mapping[logical_qubit] = physical_qubit
                   - reverse_mapping[physical_qubit] = logical_qubit

        Returns:
            Negative mean SWAP count across all instances (higher is better for LLM4AD), or None on failure.
        """
        if not self._datasets:
            return None

        swap_counts = []
        depths = []
        runtimes = []
        
        for circuit_path, data in self._datasets:
            # Instantiate a fresh router per circuit instance.
            qlosure = Qlosure(self._edges, data, with_circuit=True)
            # Bind generated heuristic per-instance to avoid cross-evaluation leakage.
            import types
            # Create a wrapper method that doesn't receive self as first argument
            # This ensures eva receives exactly the arguments it expects without self
            qlosure.generate_New_method_initial_mapping = types.MethodType(eva, qlosure)

            swaps, depth, runtime = qlosure.run(initial_mapping_method="new")
            swap_counts.append(swaps)
            depths.append(depth)
            runtimes.append(runtime)

        if not swap_counts:
            return None

        return -float(np.mean(swap_counts)), float(np.mean(depths)), -float(np.mean(runtimes))  # Negative for maximization

    


if __name__ == '__main__':

    example_code = '''
def generate_New_method_initial_mapping(distance_matrix, num_qubits, backend, backend_connections, access, dag_dependencies_count, dag2q, dag_predecessors2q):
    """
    Example: trivial identity mapping.
    """
    mapping = list(range(num_qubits))
    reverse_mapping = list(range(num_qubits))
    return mapping, reverse_mapping
'''

    evaluator = QuantumInitialMappingEvaluation(n_instance=22)  # Use just 2 circuits for quick test
    score = evaluator.evaluate_program(example_code)
    print(f"Example evaluation score (negative mean swaps): {score}")
