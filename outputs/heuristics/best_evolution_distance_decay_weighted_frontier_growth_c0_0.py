def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # --- 1. Collect logical interactions from self.access ---
    pair_weight = defaultdict(int)
    logicals_in_circuit = set()
    for _gid, qubits in self.access.items():
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
    tig = getattr(self, "temporal_interaction_graph", None)
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

    def temporal_w(a, b):
        if tig is None:
            return 0.0
        try:
            return tig[a].get(b, 0.0)
        except Exception:
            return 0.0

    # --- 2. Anchor heaviest logical edge onto densest physical edge ---
    if pair_weight and N >= 2:
        (la, lb), _w = max(
            pair_weight.items(),
            key=lambda kv: (
                kv[1],
                temporal_w(kv[0][0], kv[0][1]),
                activity.get(kv[0][0], 0) + activity.get(kv[0][1], 0),
            ),
        )

        best_edge = None
        best_score = -1.0
        seen_edges = set()
        for u, neigh in backend.items():
            for v in neigh:
                if u == v:
                    continue
                e = (u, v) if u < v else (v, u)
                if e in seen_edges:
                    continue
                seen_edges.add(e)
                deg_sum = len(backend.get(u, ())) + len(backend.get(v, ()))
                score = (centrality.get(u, 0.0) + centrality.get(v, 0.0)
                         + 1e-9 * deg_sum)
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)

        if best_edge is not None:
            pu, pv = best_edge
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

    # --- 3. Precompute initial logical-side gain to the mapped seed set ---
    gain = defaultdict(float)
    mapped_logicals = set(L for L in logicals_in_circuit if self.mapping_dict[L] != -1)
    for L in list(unmapped_logicals):
        s = 0.0
        for Lm in mapped_logicals:
            s += qig[L].get(Lm, 0)
        if s > 0:
            gain[L] = s

    # --- 4. Frontier growth with distance-decay weighted physical scoring ---
    while unmapped_logicals:
        candidates = [L for L in unmapped_logicals if L in gain and gain[L] > 0]
        if candidates:
            next_L = max(candidates,
                         key=lambda x: (gain[x], activity.get(x, 0), -x))
        else:
            remaining = [L for L in unmapped_logicals if activity.get(L, 0) > 0]
            if remaining:
                next_L = max(remaining, key=lambda x: (activity.get(x, 0), -x))
            elif unmapped_logicals:
                next_L = min(unmapped_logicals)
            else:
                break

        # gather free physical neighbors of the mapped frontier
        neighbor_pool = set()
        for P in used_phys:
            for Q in backend.get(P, ()):
                if Q not in used_phys:
                    neighbor_pool.add(Q)

        # if no mapped neighbors yet (first placement w/o anchor), seed by centrality
        chosen_P = None
        if not used_phys:
            free = [p for p in range(N) if p not in used_phys]
            if free:
                chosen_P = max(free, key=lambda p: (centrality.get(p, 0.0), -p))
        elif neighbor_pool:
            best_p_score = -float('inf')
            mapped_pairs = []
            for Lm in mapped_logicals:
                Pm = self.mapping_dict[Lm]
                if Pm < 0:
                    continue
                w = qig[next_L].get(Lm, 0)
                if w == 0:
                    continue
                tw = temporal_w(next_L, Lm)
                mapped_pairs.append((Pm, w, tw))

            for cand in neighbor_pool:
                score = 0.0
                # *** HYPOTHESIS: distance-decay weighted score (w / d^2) ***
                # super-linear penalty on shortest-path distance
                for Pm, w, tw in mapped_pairs:
                    d = dist[cand][Pm]
                    if d <= 0:
                        d = 1
                    score += (w + 0.25 * tw) / (d * d)
                score += 1e-6 * centrality.get(cand, 0.0)
                if score > best_p_score:
                    best_p_score = score
                    chosen_P = cand

        # --- 5. Fallback: most-central free physical ---
        if chosen_P is None:
            free = [p for p in range(N) if p not in used_phys]
            if not free:
                break
            chosen_P = max(free, key=lambda p: (centrality.get(p, 0.0), -p))

        assign(next_L, chosen_P)
        mapped_logicals.add(next_L)

        # incremental gain update: only newly-mapped logical adds to others
        for L in list(unmapped_logicals):
            w = qig[L].get(next_L, 0)
            if w:
                gain[L] = gain.get(L, 0.0) + w

    # --- 6. Identity-fill remaining logical ids onto free physicals ---
    free_phys = [p for p in range(N) if p not in used_phys]
    free_idx = 0
    for L in range(N):
        if self.mapping_dict[L] != -1:
            continue
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