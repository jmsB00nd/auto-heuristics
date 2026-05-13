def init_mapping(self):
    import heapq
    from collections import defaultdict, deque

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # 1. Gather logical qubits and pairwise weights.
    qig = getattr(self, "qubit_interaction_graph", None)
    pair_weight = defaultdict(float)
    logical_qubits = set()

    if qig is not None and len(qig) > 0:
        for u, nbrs in qig.items():
            logical_qubits.add(u)
            for v, w in nbrs.items():
                logical_qubits.add(v)
                if u < v:
                    pair_weight[(u, v)] += float(w)
                elif v < u:
                    pair_weight[(v, u)] += float(w)
    else:
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                logical_qubits.add(a)
                logical_qubits.add(b)
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                pair_weight[key] += 1.0

    # Include 1-qubit-only logicals too.
    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_qubits.add(q)

    # Weight lookup helper.
    def w_between(a, b):
        if qig is not None:
            return float(qig[a].get(b, 0)) if hasattr(qig[a], "get") else float(qig[a][b])
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        return pair_weight.get(key, 0.0)

    free_phys = set(range(N))
    placed = {}            # logical -> physical
    placed_logicals = []   # order of placement

    # 2. Seed on highest-frequency logical pair + most central physical edge.
    if pair_weight:
        (lA, lB), _ = max(pair_weight.items(), key=lambda kv: kv[1])

        centrality = getattr(self, "physical_centrality", {})
        def cent(p):
            try:
                return float(centrality.get(p, 0.0))
            except AttributeError:
                return float(centrality[p]) if p in centrality else 0.0

        best_edge = None
        best_score = -float("inf")
        seen_edge = set()
        for (pa, pb) in self.backend_connections:
            if pa == pb:
                continue
            key = (pa, pb) if pa < pb else (pb, pa)
            if key in seen_edge:
                continue
            seen_edge.add(key)
            s = cent(pa) + cent(pb)
            if s > best_score:
                best_score = s
                best_edge = (pa, pb)

        if best_edge is None:
            # Degenerate: no edges. Place on two most central qubits.
            sorted_phys = sorted(range(N), key=lambda p: -cent(p))
            if len(sorted_phys) >= 2:
                best_edge = (sorted_phys[0], sorted_phys[1])
            else:
                best_edge = (0, 0)

        pA, pB = best_edge
        # Order: anchor strongest logical (by activity) on most central physical.
        activity = getattr(self, "logical_activity", None)
        def act(q):
            if activity is None:
                return 0.0
            try:
                return float(activity.get(q, 0))
            except AttributeError:
                return float(activity[q]) if q in activity else 0.0

        if act(lA) < act(lB) or (act(lA) == act(lB) and cent(pA) < cent(pB)):
            lA, lB = lB, lA
        if cent(pA) < cent(pB):
            pA, pB = pB, pA

        placed[lA] = pA
        placed[lB] = pB
        free_phys.discard(pA)
        free_phys.discard(pB)
        placed_logicals.extend([lA, lB])

    # 3. Affinity: unplaced logical -> total weight to placed logicals.
    affinity = defaultdict(float)
    strongest_partner = {}  # unplaced_logical -> (best_weight, placed_logical)

    def add_affinity_from(newly_placed):
        for L in logical_qubits:
            if L in placed:
                continue
            w = w_between(L, newly_placed)
            if w > 0:
                affinity[L] += w
                cur = strongest_partner.get(L)
                if cur is None or w > cur[0] or (w == cur[0] and newly_placed < cur[1]):
                    strongest_partner[L] = (w, newly_placed)

    for pl in placed_logicals:
        add_affinity_from(pl)

    # 4. Priority-driven BFS expansion.
    def closest_free_neighbor(src_phys):
        # BFS over coupling graph; return first free physical encountered.
        if src_phys in free_phys:
            return src_phys
        visited = {src_phys}
        dq = deque([src_phys])
        while dq:
            cur = dq.popleft()
            for nb in self.backend.get(cur, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                if nb in free_phys:
                    return nb
                dq.append(nb)
        # Fallback: closest free by distance_matrix.
        best = None
        best_d = float("inf")
        for p in free_phys:
            d = self.distance_matrix[src_phys][p]
            if d < best_d:
                best_d = d
                best = p
        return best

    while affinity and free_phys:
        # Pick unplaced logical with max affinity (ties: lower id).
        L = max(affinity.keys(), key=lambda q: (affinity[q], -q))
        if affinity[L] <= 0:
            break
        partner = strongest_partner.get(L)
        if partner is None:
            del affinity[L]
            continue
        anchor_phys = placed[partner[1]]
        chosen = closest_free_neighbor(anchor_phys)
        if chosen is None:
            break

        placed[L] = chosen
        free_phys.discard(chosen)
        del affinity[L]
        strongest_partner.pop(L, None)
        add_affinity_from(L)

    # 5. Place remaining logicals that appeared in access but have no interactions.
    centrality_local = getattr(self, "physical_centrality", {})
    def cent2(p):
        try:
            return float(centrality_local.get(p, 0.0))
        except AttributeError:
            return float(centrality_local[p]) if p in centrality_local else 0.0

    remaining_logicals = sorted(
        [q for q in logical_qubits if q not in placed],
        key=lambda q: -(getattr(self, "logical_activity", {}).get(q, 0)
                        if hasattr(getattr(self, "logical_activity", {}), "get") else 0)
    )
    for L in remaining_logicals:
        if not free_phys:
            break
        # Pick most central free physical.
        chosen = max(free_phys, key=lambda p: (cent2(p), -p))
        placed[L] = chosen
        free_phys.discard(chosen)

    # 6. Identity fill for any logical id in [0, N) still unplaced.
    used_phys = set(placed.values())
    free_sorted = sorted(p for p in range(N) if p not in used_phys)
    free_iter = iter(free_sorted)
    for L in range(N):
        if L in placed:
            continue
        try:
            p = next(free_iter)
        except StopIteration:
            break
        placed[L] = p
        used_phys.add(p)

    # Safety: if any conflicts (shouldn't happen), rebuild via identity over remaining.
    final_used = set()
    for L in range(N):
        p = placed.get(L, -1)
        if p == -1 or p in final_used or not (0 <= p < N):
            placed[L] = -1
        else:
            final_used.add(p)
    leftover = [p for p in range(N) if p not in final_used]
    li = iter(leftover)
    for L in range(N):
        if placed.get(L, -1) == -1:
            try:
                placed[L] = next(li)
            except StopIteration:
                placed[L] = L  # last-resort identity

    for L in range(N):
        p = placed[L]
        self.mapping_dict[L] = p
        if 0 <= p < N:
            self.reverse_mapping_dict[p] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)