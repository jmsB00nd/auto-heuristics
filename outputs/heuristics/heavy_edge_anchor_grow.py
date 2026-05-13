def init_mapping(self):
    import heapq
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # --- 1. Collect logical interactions from self.access ---
    pair_weight = defaultdict(int)
    logicals_in_circuit = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 1:
            logicals_in_circuit.add(qubits[0])
        elif len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                logicals_in_circuit.add(a)
                continue
            logicals_in_circuit.add(a)
            logicals_in_circuit.add(b)
            key = (a, b) if a < b else (b, a)
            pair_weight[key] += 1

    qig = self.qubit_interaction_graph
    activity = self.logical_activity
    centrality = self.physical_centrality
    backend = self.backend
    dist = self.distance_matrix

    used_phys = set()
    unmapped_logicals = set(logicals_in_circuit)

    def assign(L, P):
        self.mapping_dict[L] = P
        self.reverse_mapping_dict[P] = L
        used_phys.add(P)
        unmapped_logicals.discard(L)

    # --- 2. Anchor heaviest logical edge onto densest physical edge ---
    if pair_weight and N >= 2:
        # heaviest logical edge
        (la, lb), _w = max(pair_weight.items(), key=lambda kv: (kv[1],
                            activity.get(kv[0][0], 0) + activity.get(kv[0][1], 0)))

        # densest physical edge
        best_edge = None
        best_score = -1.0
        seen_edges = set()
        for u, neigh in backend.items():
            for v in neigh:
                e = (u, v) if u < v else (v, u)
                if e in seen_edges:
                    continue
                seen_edges.add(e)
                score = (centrality.get(u, 0.0) + centrality.get(v, 0.0)
                         + 1e-9 * (len(backend.get(u, ())) + len(backend.get(v, ()))))
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)

        if best_edge is not None:
            pu, pv = best_edge
            # assign higher-activity logical to higher-centrality physical
            if activity.get(la, 0) >= activity.get(lb, 0):
                hi_L, lo_L = la, lb
            else:
                hi_L, lo_L = lb, la
            if centrality.get(pu, 0.0) >= centrality.get(pv, 0.0):
                hi_P, lo_P = pu, pv
            else:
                hi_P, lo_P = pv, pu
            assign(hi_L, hi_P)
            assign(lo_L, lo_P)

    # --- 3. Grow outward by accumulated interaction weight ---
    gain = defaultdict(float)
    for L in list(unmapped_logicals):
        s = 0.0
        for Lm in logicals_in_circuit - unmapped_logicals:
            s += qig[L].get(Lm, 0)
        if s > 0:
            gain[L] = s

    while unmapped_logicals:
        # pick next logical: max gain, tie-break on activity
        candidates = [L for L in unmapped_logicals if L in gain]
        if not candidates:
            # disconnected component — seed with highest-activity remaining
            remaining = [L for L in unmapped_logicals if activity.get(L, 0) > 0]
            if not remaining:
                break
            next_L = max(remaining, key=lambda x: activity.get(x, 0))
        else:
            next_L = max(candidates, key=lambda x: (gain[x], activity.get(x, 0)))

        # --- 4. Find best free physical neighbor of mapped set ---
        neighbor_pool = set()
        for P in used_phys:
            for Q in backend.get(P, ()):
                if Q not in used_phys:
                    neighbor_pool.add(Q)

        chosen_P = None
        if neighbor_pool:
            best_p_score = -float('inf')
            for cand in neighbor_pool:
                score = 0.0
                for Lm in logicals_in_circuit - unmapped_logicals:
                    Pm = self.mapping_dict[Lm]
                    if Pm < 0:
                        continue
                    w = qig[next_L].get(Lm, 0)
                    if w == 0:
                        continue
                    d = dist[cand][Pm]
                    score += w / (d if d > 0 else 1)
                score += 1e-6 * centrality.get(cand, 0.0)
                if score > best_p_score:
                    best_p_score = score
                    chosen_P = cand

        if chosen_P is None:
            # --- 5. Fallback: most-central free physical ---
            free = [p for p in range(N) if p not in used_phys]
            if not free:
                break
            chosen_P = max(free, key=lambda p: centrality.get(p, 0.0))

        assign(next_L, chosen_P)

        # update gains of remaining unmapped logicals
        for L in unmapped_logicals:
            w = qig[L].get(next_L, 0)
            if w:
                gain[L] = gain.get(L, 0.0) + w

    # --- 6. Identity-fill remaining logical ids onto free physicals ---
    free_phys = [p for p in range(N) if p not in used_phys]
    free_idx = 0
    for L in range(N):
        if self.mapping_dict[L] == -1:
            # prefer identity if free, else next free physical
            if L not in used_phys:
                self.mapping_dict[L] = L
                self.reverse_mapping_dict[L] = L
                used_phys.add(L)
                if L in free_phys:
                    free_phys.remove(L)
            else:
                while free_idx < len(free_phys) and free_phys[free_idx] in used_phys:
                    free_idx += 1
                if free_idx < len(free_phys):
                    P = free_phys[free_idx]
                    free_idx += 1
                    self.mapping_dict[L] = P
                    self.reverse_mapping_dict[P] = L
                    used_phys.add(P)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)