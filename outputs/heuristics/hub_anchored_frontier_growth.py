def init_mapping(self):
    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    interactions = {}
    try:
        for q1, neigh in self.qubit_interaction_graph.items():
            for q2, w in neigh.items():
                if q1 < q2 and w > 0:
                    interactions[(q1, q2)] = w
    except Exception:
        pass
    if not interactions:
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                interactions[key] = interactions.get(key, 0) + 1

    logical_neighbors = {}
    activity = {}
    for (a, b), w in interactions.items():
        logical_neighbors.setdefault(a, {})[b] = w
        logical_neighbors.setdefault(b, {})[a] = w
        activity[a] = activity.get(a, 0) + w
        activity[b] = activity.get(b, 0) + w

    try:
        for lq, act in self.logical_activity.items():
            if lq not in activity:
                activity[lq] = act
    except Exception:
        pass

    centrality = {}
    try:
        for p, c in self.physical_centrality.items():
            centrality[p] = c
    except Exception:
        for p in range(N):
            centrality[p] = 0.0

    phys_by_centrality = sorted(range(N), key=lambda p: -centrality.get(p, 0.0))

    placed_logical = set()
    used_phys = set()

    def assign(lq, pq):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        placed_logical.add(lq)
        used_phys.add(pq)

    candidates = [lq for lq in activity.keys() if 0 <= lq < N]
    if candidates:
        seed_logical = max(candidates, key=lambda lq: (activity.get(lq, 0), -lq))
        seed_phys = phys_by_centrality[0] if phys_by_centrality else 0
        assign(seed_logical, seed_phys)

    unplaced = set(lq for lq in logical_neighbors.keys() if 0 <= lq < N and lq not in placed_logical)

    while unplaced:
        best_lq = None
        best_score = -1.0
        best_act = -1
        for lq in unplaced:
            score = 0.0
            has_link = False
            for nb, w in logical_neighbors.get(lq, {}).items():
                if nb in placed_logical:
                    score += w
                    has_link = True
            if not has_link:
                continue
            act = activity.get(lq, 0)
            if score > best_score or (score == best_score and act > best_act):
                best_score = score
                best_act = act
                best_lq = lq

        if best_lq is None:
            best_lq = max(unplaced, key=lambda lq: (activity.get(lq, 0), -lq))
            free_phys = [p for p in phys_by_centrality if p not in used_phys]
            if not free_phys:
                break
            assign(best_lq, free_phys[0])
            unplaced.discard(best_lq)
            continue

        neighbor_phys_weights = []
        for nb, w in logical_neighbors.get(best_lq, {}).items():
            if nb in placed_logical:
                neighbor_phys_weights.append((self.mapping_dict[nb], w))

        best_phys = None
        best_cost = float("inf")
        best_cent = -1.0
        for p in range(N):
            if p in used_phys:
                continue
            cost = 0.0
            for npq, w in neighbor_phys_weights:
                cost += w * self.distance_matrix[p][npq]
            cent = centrality.get(p, 0.0)
            if cost < best_cost or (cost == best_cost and cent > best_cent):
                best_cost = cost
                best_cent = cent
                best_phys = p

        if best_phys is None:
            break
        assign(best_lq, best_phys)
        unplaced.discard(best_lq)

    free_phys_sorted = [p for p in phys_by_centrality if p not in used_phys]
    fp_idx = 0
    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            while fp_idx < len(free_phys_sorted) and free_phys_sorted[fp_idx] in used_phys:
                fp_idx += 1
            if fp_idx < len(free_phys_sorted):
                pq = free_phys_sorted[fp_idx]
                fp_idx += 1
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_phys.add(pq)

    if -1 in self.mapping_dict:
        remaining_phys = [p for p in range(N) if p not in used_phys]
        ri = 0
        for lq in range(N):
            if self.mapping_dict[lq] == -1 and ri < len(remaining_phys):
                pq = remaining_phys[ri]
                ri += 1
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)