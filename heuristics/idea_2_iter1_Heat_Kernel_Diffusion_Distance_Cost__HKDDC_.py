# Idea: Heat Kernel Diffusion Distance Cost (HKDDC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    import numpy as np
    from scipy.linalg import expm

    # --- Precompute heat kernel diffusion distance matrix (cached on self) ---
    if not hasattr(self, '_hkddc_cache'):
        n = self.num_qubits
        t = 1.0  # diffusion time scale; t=1 balances local vs. global topology

        # Build graph Laplacian L = D - A from hardware adjacency
        L = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            neighbors = self.backend[u]
            L[u, u] = float(len(neighbors))
            for v in neighbors:
                L[u, v] = -1.0

        # Heat kernel K_t = exp(-t * L)
        # K_t[i, j] encodes how heat "seeping" from i reaches j at time t.
        # Rows K_t[i, :] act as topological fingerprints of node i.
        K = expm(-t * L)  # shape: (n, n)

        # Diffusion distance: h_t(i, j) = || K_t[i,:] - K_t[j,:] ||_2
        # Small when i, j share many well-connected paths; large near bottlenecks.
        # Broadcasting: (n, 1, n) - (1, n, n) → (n, n, n), then norm over axis=2
        diff = K[:, np.newaxis, :] - K[np.newaxis, :, :]   # (n, n, n)
        diff_dist = np.linalg.norm(diff, axis=2)            # (n, n)

        # Normalise so diffusion distances are on a comparable scale to hop counts.
        # We map [0, max] → [0, max_hop], preserving relative topology.
        max_hop = float(np.max(self.distance_matrix))
        d_max = diff_dist.max()
        if d_max > 1e-12:
            diff_dist = diff_dist * (max_hop / d_max)

        self._hkddc_cache = diff_dist

    diff_dist = self._hkddc_cache

    # --- Cost computation (same structure as baseline; distance metric replaced) ---
    W = 1.0
    front_layer_size  = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # Front layer: weight by criticality, use diffusion distance
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2  = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps    = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * diff_dist[Q1][Q2]

    # Extended (lookahead) layer: depth-discounted diffusion distance
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2       = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps         = self.dag_dependencies_count[g]
        e_distance  += (deps + 1) * diff_dist[Q1][Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size
        + W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H