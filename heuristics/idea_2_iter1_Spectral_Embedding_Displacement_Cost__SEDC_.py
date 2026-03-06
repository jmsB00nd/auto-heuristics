# Idea: Spectral Embedding Displacement Cost (SEDC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    import numpy as np

    # === Build & Cache Spectral Embedding ===
    if not hasattr(self, '_sedc_embedding'):
        n = self.num_qubits
        k = min(8, n - 1)   # number of non-trivial eigenvectors to keep

        # Unnormalized graph Laplacian  L = D - A
        L = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in self.backend.items():
            L[u, u] = float(len(neighbors))
            for v in neighbors:
                L[u, v] = -1.0

        # eigh is exact + efficient for real symmetric matrices
        # eigenvalues sorted ascending: λ_0 ≈ 0 (trivial), λ_1 … λ_k (non-trivial)
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Skip the trivial zero eigenvalue (index 0); take next k columns
        lam = eigenvalues[1:k + 1]          # shape (k,)
        U   = eigenvectors[:, 1:k + 1]      # shape (n, k)

        # Scale each axis by 1/sqrt(λ_i): qubit pairs that are far apart
        # along low-frequency eigenvectors are penalised more (harder to route)
        scale = 1.0 / np.sqrt(np.maximum(lam, 1e-10))
        embedding = U * scale[np.newaxis, :]  # shape (n, k)

        self._sedc_embedding = embedding

    emb = self._sedc_embedding

    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # === Front Layer: squared spectral displacement ===
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 == -1 or Q2 == -1:
            continue
        deps   = self.dag_dependencies_count[g]
        diff   = emb[Q1] - emb[Q2]
        sq_dist = float(np.dot(diff, diff))
        f_cost += (deps + 1) * sq_dist

    # === Extended Layer: discounted squared spectral displacement ===
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 == -1 or Q2 == -1:
            continue
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps         = self.dag_dependencies_count[g]
        diff         = emb[Q1] - emb[Q2]
        sq_dist      = float(np.dot(diff, diff))
        e_cost += (deps + 1) * sq_dist / layer_factor

    H = max_decay * (
        f_cost / front_layer_size
        + (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H