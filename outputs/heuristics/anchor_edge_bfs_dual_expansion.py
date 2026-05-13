def init_mapping(self):
    import heapq
    from collections import defaultdict, deque

    N = self.num_qubits

    pair_weight = defaultdict(float)
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_qubits.add(a)
            logical_qubits.add(b)
            key = (a, b) if a < b else (b, a)
            qig_w = 0
            try:
                qig_w = self.qubit_interaction_graph[a].get(b, 0)
            except Exception:
                qig_w = 0
            pair_weight[key] += qig_w if qig_w > 0 else 1
        elif len(qubits) == 1:
            logical_qubits.add(qubits[0])

    logical_neighbors = defaultdict(dict)
    for (a, b), w in pair_weight.items():
        logical_neighbors[a][b] = w
        logical_neighbors[b][a] = w

    activity = {}
    for q in logical_qubits:
        try:
            activity[q] = float(self.logical_activity.get(q, 0))
        except Exception:
            activity[q] = 0.0
        if activity[q] == 0.0:
            activity[q] = sum(logical_neighbors[q].values())

    centrality = {}
    for p in range(N):
        try:
            centrality[p] = float(self.physical_centrality.get(p, 0.0))
        except Exception:
            centrality[p] = 0.0

    phys_degree = {p: len(self.backend.get(p, set())) for p in range(N)}

    mapping = [-1] * N
    reverse = [-1] * N
    used_physical = set()
    placed_logical = set()

    def place(lq, pq):
        mapping[lq] = pq
        reverse[pq] = lq
        used_physical.add(pq)
        placed_logical.add(lq)

    if pair_weight and self.backend_connections:
        seed_logical = max(pair_weight.items(), key=lambda kv: (kv[1], activity.get(kv[0][0], 0) + activity.get(kv[0][1], 0)))[0]
        la, lb = seed_logical
        if activity.get(la, 0) < activity.get(lb, 0):
            la, lb = lb, la

        seen_edges = set()
        best_phys_edge = None
        best_score = None
        for (pa, pb) in self.backend_connections:
            key = (pa, pb) if pa < pb else (pb, pa)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            score = (centrality.get(pa, 0.0) + centrality.get(pb, 0.0),
                     phys_degree.get(pa, 0) + phys_degree.get(pb, 0))
            if best_score is None or score > best_score:
                best_score = score
                best_phys_edge = (pa, pb)

        if best_phys_edge is not None:
            pa, pb = best_phys_edge
            if centrality.get(pa, 0.0) < centrality.get(pb, 0.0):
                pa, pb = pb, pa
            place(la, pa)
            place(lb, pb)

    if placed_logical:
        def frontier_score(lq):
            s = 0.0
            for nq, w in logical_neighbors[lq].items():
                if nq in placed_logical:
                    s += w
            return s

        active = [lq for lq in logical_qubits if lq not in placed_logical]
        while active:
            best_lq = None
            best_val = -1.0
            for lq in active:
                fs = frontier_score(lq)
                if fs > best_val or (fs == best_val and activity.get(lq, 0) > activity.get(best_lq, -1) if best_lq is not None else False):
                    best_val = fs
                    best_lq = lq
            if best_lq is None or best_val <= 0:
                break

            anchor_phys = []
            for nq, w in logical_neighbors[best_lq].items():
                if nq in placed_logical:
                    anchor_phys.append((mapping[nq], w))

            candidate_scores = {}
            for ap, w in anchor_phys:
                for neighbor in self.backend.get(ap, set()):
                    if neighbor in used_physical:
                        continue
                    candidate_scores[neighbor] = candidate_scores.get(neighbor, 0.0) + w

            chosen_phys = None
            if candidate_scores:
                chosen_phys = max(candidate_scores.items(),
                                  key=lambda kv: (kv[1], centrality.get(kv[0], 0.0)))[0]
            else:
                best_dist = None
                for p in range(N):
                    if p in used_physical:
                        continue
                    d_acc = 0.0
                    for ap, w in anchor_phys:
                        try:
                            d = self.distance_matrix[ap][p]
                        except Exception:
                            d = N
                        d_acc += w * d
                    score = (-d_acc, centrality.get(p, 0.0))
                    if best_dist is None or score > best_dist:
                        best_dist = score
                        chosen_phys = p

            if chosen_phys is None:
                break

            place(best_lq, chosen_phys)
            active = [lq for lq in logical_qubits if lq not in placed_logical]

    remaining_logicals = [lq for lq in logical_qubits if lq not in placed_logical]
    remaining_logicals.sort(key=lambda q: -activity.get(q, 0))
    free_physicals_sorted = sorted([p for p in range(N) if p not in used_physical],
                                   key=lambda p: -centrality.get(p, 0.0))
    fp_iter = iter(free_physicals_sorted)
    for lq in remaining_logicals:
        try:
            pq = next(fp_iter)
        except StopIteration:
            break
        place(lq, pq)

    free_physicals = [p for p in range(N) if p not in used_physical]
    fp_idx = 0
    for lq in range(N):
        if mapping[lq] == -1:
            if lq not in used_physical:
                place(lq, lq)
            else:
                while fp_idx < len(free_physicals) and free_physicals[fp_idx] in used_physical:
                    fp_idx += 1
                if fp_idx < len(free_physicals):
                    place(lq, free_physicals[fp_idx])
                    fp_idx += 1

    if -1 in mapping:
        free_physicals = [p for p in range(N) if p not in used_physical]
        fi = 0
        for lq in range(N):
            if mapping[lq] == -1 and fi < len(free_physicals):
                place(lq, free_physicals[fi])
                fi += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)