def init_mapping(self):
    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # Gather logical qubits appearing in 2-qubit gates and their pair weights
    pair_weight = {}
    logicals_in_circuit = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logicals_in_circuit.add(a)
            logicals_in_circuit.add(b)
            key = (a, b) if a < b else (b, a)
            pair_weight[key] = pair_weight.get(key, 0) + 1

    # Per-logical activity via QIG (fallback to recomputed sums)
    activity = {}
    for L in logicals_in_circuit:
        try:
            w = self.logical_activity[L]
        except Exception:
            w = 0
        if not w:
            w = sum(self.qubit_interaction_graph[L].values()) if L in self.qubit_interaction_graph else 0
        activity[L] = w

    def pair_w(u, v):
        if u == v:
            return 0
        try:
            w = self.qubit_interaction_graph[u][v]
            if w:
                return w
        except Exception:
            pass
        key = (u, v) if u < v else (v, u)
        return pair_weight.get(key, 0)

    # Centrality lookup with safe default
    def centrality(p):
        try:
            return self.physical_centrality.get(p, 0.0)
        except Exception:
            return 0.0

    # Priority order: highest-activity logicals first; ties broken by id
    ordered_logicals = sorted(
        logicals_in_circuit,
        key=lambda L: (-activity.get(L, 0), L),
    )

    used_phys = set()

    # Seed: most central physical for the first (highest-activity) logical
    if ordered_logicals:
        seed_logical = ordered_logicals[0]
        seed_phys = max(range(N), key=lambda p: (centrality(p), -p))
        self.mapping_dict[seed_logical] = seed_phys
        self.reverse_mapping_dict[seed_phys] = seed_logical
        used_phys.add(seed_phys)

    placed = set(ordered_logicals[:1])

    # Weiszfeld-style incremental 1-median placement
    for L in ordered_logicals[1:]:
        neighbor_weights = []
        for n in placed:
            w = pair_w(L, n)
            if w > 0:
                neighbor_weights.append((self.mapping_dict[n], w))

        best_phys = -1
        if neighbor_weights:
            best_cost = None
            best_centr = None
            for p in range(N):
                if p in used_phys:
                    continue
                cost = 0.0
                row = self.distance_matrix[p]
                for np_, w in neighbor_weights:
                    cost += w * row[np_]
                c = centrality(p)
                if (best_cost is None) or (cost < best_cost) or (cost == best_cost and c > best_centr):
                    best_cost = cost
                    best_centr = c
                    best_phys = p
        else:
            best_centr = None
            for p in range(N):
                if p in used_phys:
                    continue
                c = centrality(p)
                if best_centr is None or c > best_centr:
                    best_centr = c
                    best_phys = p

        if best_phys < 0:
            for p in range(N):
                if p not in used_phys:
                    best_phys = p
                    break

        self.mapping_dict[L] = best_phys
        self.reverse_mapping_dict[best_phys] = L
        used_phys.add(best_phys)
        placed.add(L)

    # Back-fill remaining logical ids with unused physicals (identity-preferring)
    remaining_logicals = [L for L in range(N) if self.mapping_dict[L] == -1]
    remaining_phys = [p for p in range(N) if p not in used_phys]
    # Prefer identity assignment when possible
    identity_taken = set()
    for L in remaining_logicals:
        if L in remaining_phys and L not in identity_taken:
            self.mapping_dict[L] = L
            self.reverse_mapping_dict[L] = L
            used_phys.add(L)
            identity_taken.add(L)
    remaining_phys = [p for p in range(N) if p not in used_phys]
    rp_iter = iter(remaining_phys)
    for L in range(N):
        if self.mapping_dict[L] == -1:
            p = next(rp_iter)
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)