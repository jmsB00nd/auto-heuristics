def init_mapping(self):
    import numpy as np
    from scipy.linalg import orthogonal_procrustes

    n = self.num_qubits

    interaction_weight = {}
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits.add(q1)
            logical_qubits.add(q2)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] = interaction_weight.get(key, 0) + 1

    if not logical_qubits or n <= 2:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    hw_adj = np.zeros((n, n))
    for u, v in self.backend_connections:
        hw_adj[u][v] = 1.0
        hw_adj[v][u] = 1.0

    int_adj = np.zeros((n, n))
    for (u, v), w in interaction_weight.items():
        int_adj[u][v] = float(w)
        int_adj[v][u] = float(w)

    def spectral_embed(adj, k):
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.zeros(n)
        mask = degree > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
        D = np.diag(d_inv_sqrt)
        L = np.eye(n) - D @ adj @ D
        L = (L + L.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(L)
        return eigvecs[:, 1:k + 1]

    k = min(10, n - 1)

    hw_embed = spectral_embed(hw_adj, k)
    int_embed = spectral_embed(int_adj, k)

    actual_k = min(hw_embed.shape[1], int_embed.shape[1])
    if actual_k < 1:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    hw_embed = hw_embed[:, :actual_k]
    int_embed = int_embed[:, :actual_k]

    R, _ = orthogonal_procrustes(int_embed, hw_embed)
    aligned = int_embed @ R

    logical_degree = {}
    for q in logical_qubits:
        logical_degree[q] = int_adj[q].sum()
    sorted_logical = sorted(logical_qubits, key=lambda q: -logical_degree[q])

    mapping = [0] * n
    used_physical = set()
    assigned_logical = set()

    for lq in sorted_logical:
        lq_coord = aligned[lq]
        best_pq = -1
        best_dist = float('inf')
        for pq in range(n):
            if pq in used_physical:
                continue
            dist = float(np.linalg.norm(lq_coord - hw_embed[pq]))
            if dist < best_dist:
                best_dist = dist
                best_pq = pq
        mapping[lq] = best_pq
        used_physical.add(best_pq)
        assigned_logical.add(lq)

    remaining_physical = sorted(set(range(n)) - used_physical)
    remaining_logical = sorted(set(range(n)) - assigned_logical)
    for lq, pq in zip(remaining_logical, remaining_physical):
        mapping[lq] = pq

    reverse_mapping = [0] * n
    for lq in range(n):
        reverse_mapping[mapping[lq]] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)