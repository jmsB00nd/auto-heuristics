# Idea: k-Step Random Walk Distance Cost (KRWDC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    import numpy as np

    K = 2  # k-step random walk parameter

    # Build and cache the k-step random walk L2 distance matrix once per topology
    if not hasattr(self, '_krw_dist_matrix'):
        n = self.num_qubits

        # Build adjacency matrix from hardware graph
        A = np.zeros((n, n), dtype=float)
        for u in range(n):
            for v in self.backend[u]:
                A[u][v] = 1.0

        # Row-stochastic transition matrix: P = D^{-1} A
        degrees = A.sum(axis=1)
        D_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
        P = A * D_inv[:, np.newaxis]

        # k-step transition matrix P^k
        Pk = np.linalg.matrix_power(P, K)

        # Pairwise L2 distance between rows of P^k
        # ||P^k[i] - P^k[j]||_2 measures divergence of k-hop neighborhoods
        # Broadcasting: (n,1,n) - (1,n,n) -> (n,n,n), then norm over last axis
        diff = Pk[:, np.newaxis, :] - Pk[np.newaxis, :, :]  # (n, n, n)
        self._krw_dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))  # (n, n)

    krw_dist = self._krw_dist_matrix

    W = 1.0
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # Front layer: weighted sum of k-step RW distances with criticality
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * krw_dist[Q1, Q2]

    # Extended layer: discounted by lookahead depth and criticality
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        e_distance += (deps + 1) * krw_dist[Q1, Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size
        + W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H