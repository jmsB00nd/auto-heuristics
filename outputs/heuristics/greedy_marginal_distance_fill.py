def init_mapping(self):
    N = self.num_qubits
    N_phys = max(N - 1, 0)

    qig = self.qubit_interaction_graph
    activity = self.logical_activity
    centrality = self.physical_centrality
    dist = self.distance_matrix

    active_logicals = set()
    for q1, nbrs in qig.items():
        for q2, w in nbrs.items():
            if w > 0:
                active_logicals.add(q1)
                active_logicals.add(q2)

    mapping = [-1] * N
    placed = set()
    unplaced = set(active_logicals)
    free_physical = set(range(N_phys))

    def central_pick(free_set):
        return max(free_set, key=lambda p: (centrality.get(p, 0.0), -p))

    if unplaced and free_physical:
        seed_L = max(unplaced, key=lambda q: (activity.get(q, 0), -q))
        seed_P = central_pick(free_physical)
        mapping[seed_L] = seed_P
        placed.add(seed_L)
        unplaced.discard(seed_L)
        free_physical.discard(seed_P)

        while unplaced and free_physical:
            best = None
            best_cost = float('inf')
            connected_exists = False
            for L in unplaced:
                placed_nbrs = [(L2, qig[L][L2]) for L2 in qig.get(L, {}) if L2 in placed and qig[L][L2] > 0]
                if not placed_nbrs:
                    continue
                connected_exists = True
                for P in free_physical:
                    cost = 0.0
                    for L2, w in placed_nbrs:
                        cost += w * dist[P][mapping[L2]]
                        if cost >= best_cost:
                            break
                    if cost < best_cost:
                        best_cost = cost
                        best = (L, P)

            if not connected_exists:
                L = max(unplaced, key=lambda q: (activity.get(q, 0), -q))
                P = central_pick(free_physical)
                best = (L, P)

            L, P = best
            mapping[L] = P
            placed.add(L)
            unplaced.discard(L)
            free_physical.discard(P)

    used = set(p for p in mapping if p != -1)
    available = sorted(set(range(N)) - used)
    ai = 0
    for L in range(N):
        if mapping[L] == -1:
            if L not in used:
                mapping[L] = L
                used.add(L)
                available = [a for a in available if a != L]
            else:
                while ai < len(available) and available[ai] in used:
                    ai += 1
                if ai < len(available):
                    P = available[ai]
                    mapping[L] = P
                    used.add(P)
                    ai += 1

    reverse = [-1] * N
    for L in range(N):
        P = mapping[L]
        if 0 <= P < N:
            reverse[P] = L

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    if self.use_isl:
        try:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        except Exception:
            self.isl_mapping = None

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)