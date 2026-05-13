def init_mapping(self):
    import numpy as np

    N = int(self.num_qubits)

    logical_set = set()
    for _gid, qubits in self.access.items():
        for q in qubits:
            logical_set.add(int(q))

    if N <= 0:
        self.mapping_dict = []
        self.reverse_mapping_dict = []
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    if not logical_set:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    features = {}
    for q in logical_set:
        v = np.zeros(N, dtype=float)
        if q in self.qubit_interaction_graph:
            for nb, w in self.qubit_interaction_graph[q].items():
                nb_i = int(nb)
                if 0 <= nb_i < N:
                    v[nb_i] = float(w)
        nrm = float(np.linalg.norm(v))
        if nrm > 0.0:
            v = v / nrm
        features[q] = v

    rng = np.random.default_rng(0xC0DE)
    W = rng.uniform(0.0, 1e-3, size=(N, N))

    logical_list = sorted(logical_set)
    raw_act = np.array(
        [max(float(self.logical_activity.get(q, 0)), 1.0) for q in logical_list],
        dtype=float,
    )
    probs = raw_act / raw_act.sum()

    try:
        D = np.asarray(self.distance_matrix, dtype=float)
        if D.shape != (N, N):
            D = np.zeros((N, N), dtype=float)
    except Exception:
        D = np.zeros((N, N), dtype=float)

    max_d = float(D.max()) if D.size > 0 else 1.0
    initial_radius = max(1.0, max_d / 2.0)
    final_radius = 0.5
    initial_lr = 0.5
    final_lr = 0.01
    num_iter = max(100, 8 * len(logical_list))

    for t in range(num_iter):
        frac = t / max(1, num_iter - 1)
        lr = initial_lr * (final_lr / initial_lr) ** frac
        radius = initial_radius * (final_radius / initial_radius) ** frac

        idx = int(rng.choice(len(logical_list), p=probs))
        q = logical_list[idx]
        x = features[q]

        diffs = W - x[None, :]
        sqd = np.einsum("ij,ij->i", diffs, diffs)
        bmu = int(np.argmin(sqd))

        d_to_bmu = D[bmu]
        h = np.exp(-(d_to_bmu * d_to_bmu) / (2.0 * radius * radius + 1e-12))
        h = np.where(d_to_bmu <= radius + 1e-9, h, 0.0)
        W += (lr * h)[:, None] * (x[None, :] - W)

    mapping = [-1] * N
    rev = [-1] * N
    used = [False] * N

    centrality = getattr(self, "physical_centrality", {}) or {}
    cent_arr = np.array([float(centrality.get(p, 0.0)) for p in range(N)], dtype=float)

    order_logicals = sorted(
        logical_list,
        key=lambda q: -float(self.logical_activity.get(q, 0)),
    )

    for q in order_logicals:
        x = features[q]
        diffs = W - x[None, :]
        sqd = np.einsum("ij,ij->i", diffs, diffs)
        keys = sqd - 1e-9 * cent_arr
        order = np.argsort(keys, kind="stable")
        for p in order:
            p_i = int(p)
            if not used[p_i]:
                mapping[q] = p_i
                rev[p_i] = q
                used[p_i] = True
                break

    unused = [p for p in range(N) if not used[p]]
    ui = 0
    for q in range(N):
        if mapping[q] == -1:
            if ui < len(unused):
                p_i = unused[ui]
                ui += 1
                mapping[q] = p_i
                rev[p_i] = q

    self.mapping_dict = mapping
    self.reverse_mapping_dict = rev

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)