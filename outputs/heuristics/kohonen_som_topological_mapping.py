def init_mapping(self):
    import numpy as np

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    logical_qubits = set()
    for _gate_id, qubits in self.access.items():
        for q in qubits:
            if 0 <= q < N:
                logical_qubits.add(int(q))

    if not logical_qubits:
        for i in range(N):
            self.mapping_dict[i] = i
            self.reverse_mapping_dict[i] = i
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # --- Build logical input vectors x_q from QIG rows ---
    x = np.zeros((N, N), dtype=float)
    for q in logical_qubits:
        row = self.qubit_interaction_graph.get(q, {})
        for k, w_qk in row.items():
            if 0 <= k < N:
                x[q, k] = float(w_qk)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    x_norm = x / norms

    # --- Hardware lattice distances ---
    hw_dist = np.array(self.distance_matrix, dtype=float)
    max_d = float(hw_dist.max()) if hw_dist.size > 0 and hw_dist.max() > 0 else 1.0

    # --- Initialize neuron weight vectors ---
    rng = np.random.default_rng(seed=1234567)
    w = rng.standard_normal((N, N)) * 0.01

    # --- SOM training schedule ---
    activity = self.logical_activity
    logical_list = sorted(logical_qubits, key=lambda q: -activity.get(q, 0))

    n_epochs = 6
    eta0, eta_f = 0.6, 0.02
    sigma0 = max(1.0, max_d)
    sigma_f = 0.5

    total_iters = max(1, n_epochs * len(logical_list))
    it = 0
    for _epoch in range(n_epochs):
        order = list(logical_list)
        rng.shuffle(order)
        for q in order:
            t = it / total_iters
            eta = eta0 * ((eta_f / eta0) ** t)
            sigma = sigma0 * ((sigma_f / sigma0) ** t)
            sigma_sq = max(sigma * sigma, 1e-9)

            xq = x_norm[q]
            diffs = w - xq
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            bmu = int(np.argmin(d2))

            h = np.exp(-(hw_dist[bmu] ** 2) / (2.0 * sigma_sq))
            w += (eta * h)[:, None] * (xq - w)
            it += 1

    # --- Assignment: activity-ordered greedy on BMU ranking ---
    used_physical = set()
    for q in sorted(logical_qubits, key=lambda q: -activity.get(q, 0)):
        xq = x_norm[q]
        diffs = w - xq
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        ranking = np.argsort(d2)
        chosen = -1
        for p in ranking:
            p_int = int(p)
            if p_int not in used_physical:
                chosen = p_int
                break
        if chosen < 0:
            for p_int in range(N):
                if p_int not in used_physical:
                    chosen = p_int
                    break
        self.mapping_dict[q] = chosen
        self.reverse_mapping_dict[chosen] = q
        used_physical.add(chosen)

    # --- Back-fill idle logical ids with remaining free physicals ---
    remaining_physical = [p for p in range(N) if p not in used_physical]
    remaining_logical = [l for l in range(N) if self.mapping_dict[l] == -1]
    for l, p in zip(remaining_logical, remaining_physical):
        self.mapping_dict[l] = p
        self.reverse_mapping_dict[p] = l
        used_physical.add(p)

    for l in range(N):
        if self.mapping_dict[l] == -1:
            for p in range(N):
                if p not in used_physical:
                    self.mapping_dict[l] = p
                    self.reverse_mapping_dict[p] = l
                    used_physical.add(p)
                    break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)