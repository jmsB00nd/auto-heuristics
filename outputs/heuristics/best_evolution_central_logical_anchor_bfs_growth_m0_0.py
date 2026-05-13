def init_mapping(self):
    from collections import defaultdict
    import heapq

    N = self.num_qubits
    backend = self.backend
    bconn = self.backend_connections
    centrality = self.physical_centrality

    # ---- build QIG + ordered 2q gate sequence ----
    qig = defaultdict(lambda: defaultdict(int))
    logical_set = set()
    gate_seq = []
    for gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_set.add(a); logical_set.add(b)
            if a != b:
                qig[a][b] += 1
                qig[b][a] += 1
                gate_seq.append((gid, a, b))
        elif len(qubits) == 1:
            logical_set.add(qubits[0])

    try:
        gate_seq.sort(key=lambda x: x[0])
    except Exception:
        pass

    log_deg = {q: len(qig[q]) for q in logical_set}
    log_wgt = {q: sum(qig[q].values()) for q in logical_set}
    phys_deg = {p: len(backend[p]) for p in range(N)}

    # ---- HYPOTHESIS: pick the highest-traffic logical qubit as the anchor ----
    # logical_centrality = total interaction weight + degree (tiebreak)
    # The hypothesis claims this anchored-at-core seed beats ASAP-depth seeds
    # because it minimizes average traversal distance for its many neighbors.
    def logical_centrality_key(q):
        return (-log_wgt[q], -log_deg[q], q)

    logicals_by_central = sorted(logical_set, key=logical_centrality_key)

    phys_by_central = sorted(
        range(N),
        key=lambda p: (-centrality.get(p, 0.0), -phys_deg[p], p),
    )

    # ---- BFS-growth ordering: visit logicals in the order they are reached
    # while expanding outward from the seed (the most-central logical). Within
    # each frontier, prefer the neighbor with the heaviest edge to the placed
    # subgraph, then highest overall weight.
    def build_growth_order(seed):
        order = [seed]
        visited = {seed}
        # priority queue: (-edge_weight_to_placed, -log_wgt, -log_deg, q)
        # cumulative weight to placed subgraph
        weight_to_placed = defaultdict(int)
        for v, w in qig[seed].items():
            weight_to_placed[v] += w
        heap = []
        for v in qig[seed]:
            if v in logical_set and v != seed:
                heapq.heappush(
                    heap,
                    (-weight_to_placed[v], -log_wgt.get(v, 0), -log_deg.get(v, 0), v),
                )
        in_heap = set(qig[seed].keys()) & logical_set
        in_heap.discard(seed)
        while heap:
            negw, _nw, _nd, u = heapq.heappop(heap)
            if u in visited:
                continue
            # stale check: only accept entries with current best weight
            if -negw != weight_to_placed[u]:
                continue
            visited.add(u)
            order.append(u)
            for v, w in qig[u].items():
                if v in visited or v not in logical_set:
                    continue
                weight_to_placed[v] += w
                heapq.heappush(
                    heap,
                    (-weight_to_placed[v], -log_wgt.get(v, 0), -log_deg.get(v, 0), v),
                )
        # Append any logicals disconnected from the seed component, sorted by
        # centrality so they are still placed core-first.
        for u in logicals_by_central:
            if u not in visited:
                visited.add(u)
                order.append(u)
        return order

    # candidate generator: free physicals adjacent to already-placed neighbors,
    # otherwise top-central free physicals.
    def candidates(u, phi, used):
        cand_set = set()
        for v in qig[u]:
            if v in phi:
                for x in backend[phi[v]]:
                    if x not in used:
                        cand_set.add(x)
        if not cand_set:
            limit = max(8, N // 4)
            for p in phys_by_central:
                if p not in used:
                    cand_set.add(p)
                    if len(cand_set) >= limit:
                        break
        scored = []
        for p in cand_set:
            adj_bonus = 0
            for v, w in qig[u].items():
                if v in phi:
                    pv = phi[v]
                    if (p, pv) in bconn or (pv, p) in bconn:
                        adj_bonus += w
            scored.append((-adj_bonus, -centrality.get(p, 0.0), -phys_deg[p], p))
        scored.sort()
        return [t[3] for t in scored]

    def feasible(u, p, phi, used):
        if p in used:
            return False
        if phys_deg[p] < log_deg[u]:
            unmapped_nbrs = sum(1 for v in qig[u] if v not in phi)
            mapped_nbrs = log_deg[u] - unmapped_nbrs
            if phys_deg[p] < mapped_nbrs:
                return False
            free_phys_nbrs = sum(1 for x in backend[p] if x not in used)
            if unmapped_nbrs > 0 and free_phys_nbrs < unmapped_nbrs - 1:
                return False
        return True

    best = {"score": -1, "phi": {}}

    # try the top central-logical seeds, each anchored at the top central physicals
    seed_logical_count = min(len(logicals_by_central), 3) if logicals_by_central else 0
    seed_phys_count = min(len(phys_by_central), 3) if phys_by_central else 0

    for li in range(max(1, seed_logical_count)):
        if not logicals_by_central:
            break
        seed_logical = logicals_by_central[li]
        growth_order = build_growth_order(seed_logical)
        if not growth_order:
            continue

        budget = [max(2000, 50 * max(1, len(growth_order)) * max(1, N))]

        def backtrack(idx, phi, used, partial_score, order=growth_order, budget=budget):
            if budget[0] <= 0:
                return
            budget[0] -= 1
            if idx == len(order):
                if partial_score > best["score"]:
                    best["score"] = partial_score
                    best["phi"] = dict(phi)
                return
            u = order[idx]
            cands = candidates(u, phi, used)
            if not cands:
                if partial_score > best["score"]:
                    best["score"] = partial_score
                    best["phi"] = dict(phi)
                return
            for p in cands:
                if not feasible(u, p, phi, used):
                    continue
                gain = 0
                for v, w in qig[u].items():
                    if v in phi:
                        pv = phi[v]
                        if (p, pv) in bconn or (pv, p) in bconn:
                            gain += w
                phi[u] = p
                used.add(p)
                backtrack(idx + 1, phi, used, partial_score + gain)
                del phi[u]
                used.discard(p)
                if budget[0] <= 0:
                    return

        # anchor the central logical seed at top-central physicals
        for pi in range(max(1, seed_phys_count)):
            if budget[0] <= 0:
                break
            seed_phys = phys_by_central[pi]
            if not feasible(seed_logical, seed_phys, {}, set()):
                continue
            phi0 = {seed_logical: seed_phys}
            used0 = {seed_phys}
            backtrack(1, phi0, used0, 0)

    phi = dict(best["phi"])

    # ---- fallback: structure-aware mapping for any unplaced logicals ----
    if any(q not in phi for q in logical_set):
        try:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            fb_map, _fb_rev = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, self.num_qubits
            )
            used_now = set(phi.values())
            for L in logicals_by_central:
                if L in phi:
                    continue
                cand = fb_map[L] if L < len(fb_map) else None
                if cand is not None and cand not in used_now:
                    phi[L] = cand
                    used_now.add(cand)
        except Exception:
            pass

    # ---- materialize into list-based mapping ----
    used_phys = set(phi.values())
    free_phys = [p for p in phys_by_central if p not in used_phys]

    mapping_list = [-1] * N
    for L, P in phi.items():
        if 0 <= L < N and 0 <= P < N:
            mapping_list[L] = P

    free_logicals = [L for L in range(N) if mapping_list[L] == -1]
    fp_iter = iter(free_phys)
    for L in free_logicals:
        try:
            mapping_list[L] = next(fp_iter)
        except StopIteration:
            break

    if any(x == -1 for x in mapping_list):
        used_phys2 = set(p for p in mapping_list if p != -1)
        leftover = [p for p in range(N) if p not in used_phys2]
        li2 = iter(leftover)
        for L in range(N):
            if mapping_list[L] == -1:
                try:
                    mapping_list[L] = next(li2)
                except StopIteration:
                    for p in range(N):
                        if p not in used_phys2:
                            mapping_list[L] = p
                            used_phys2.add(p)
                            break

    reverse_list = [-1] * N
    for L in range(N):
        P = mapping_list[L]
        if 0 <= P < N:
            reverse_list[P] = L

    if any(x == -1 for x in reverse_list):
        used_log = set(L for L in reverse_list if L != -1)
        free_log = [L for L in range(N) if L not in used_log]
        fl_iter = iter(free_log)
        for P in range(N):
            if reverse_list[P] == -1:
                try:
                    reverse_list[P] = next(fl_iter)
                except StopIteration:
                    break

    self.mapping_dict = mapping_list
    self.reverse_mapping_dict = reverse_list

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)