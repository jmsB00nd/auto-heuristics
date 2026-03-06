# Idea: Effective Resistance (Commute Time) Routing Cost (ERCRC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on multiplier_n75__1308CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    import numpy as np

    # Build & cache the effective resistance matrix (computed once per topology)
    if not hasattr(self, '_eff_resistance_cache'):
        n = self.num_qubits

        # Combinatorial graph Laplacian  L = D - A
        L = np.zeros((n, n), dtype=float)
        for u, neighbors in self.backend.items():
            for v in neighbors:
                L[u][u] += 1.0
                L[u][v] -= 1.0

        # Moore-Penrose pseudoinverse of L
        L_plus = np.linalg.pinv(L)

        # Effective resistance: R_eff(i,j) = L+[i,i] + L+[j,j] - 2*L+[i,j]
        # This equals (e_i - e_j)^T L^+ (e_i - e_j) by definition
        diag = np.diag(L_plus)
        R_eff = diag[:, None] + diag[None, :] - 2.0 * L_plus
        np.fill_diagonal(R_eff, 0.0)
        self._eff_resistance_cache = R_eff

    R_eff = self._eff_resistance_cache

    W = 0.5
    front_layer_size  = max(len(self.front_layer), 1)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front-layer cost ---
    # Bottleneck paths carry higher R_eff, so swaps that route through
    # congested regions are naturally penalised more than hop-count alone.
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        criticality = self.dag_dependencies_count[g] + 1
        f_cost += criticality * R_eff[Q1][Q2]

    # --- Extended-layer (lookahead) cost ---
    # Discount by lookahead depth so near-future gates dominate.
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        depth = self.extended_layer_index.get(g, 0) + 1
        criticality = self.dag_dependencies_count[g] + 1
        e_cost += criticality * R_eff[Q1][Q2] / depth

    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H