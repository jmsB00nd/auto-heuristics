def init_mapping(self):
    import heapq
    import time
    from collections import defaultdict

    N = self.num_qubits
    dist = self.distance_matrix

    weights = defaultdict(int)
    logical_set = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_set.add(a); logical_set.add(b)
            if a != b:
                key = (a, b) if a < b else (b, a)
                weights[key] += 1
        elif len(qubits) == 1:
            logical_set.add(qubits[0])

    activity = defaultdict(int)
    for (a, b), w in weights.items():
        activity[a] += w
        activity[b] += w

    active = sorted([l for l in logical_set if activity[l] > 0 and 0 <= l < N],
                    key=lambda x: -activity[x])
    K = min(len(active), N)
    L = active[:K]
    L_index = {l: i for i, l in enumerate(L)}
    n_active = len(L)

    logical_adj = defaultdict(list)
    for (a, b), w in weights.items():
        if a in L_index and b in L_index:
            logical_adj[a].append((b, w))
            logical_adj[b].append((a, w))

    centrality = getattr(self, 'physical_centrality', None) or {}
    physicals_sorted = sorted(range(N), key=lambda p: -centrality.get(p, 0.0))

    mapping = [-1] * N
    used_phys = set()
    best_assignment = None

    if n_active > 0:
        min_edge_dist = 1
        branching = max(4, min(N, 10))
        node_budget = 20000
        time_budget = 2.0
        start_t = time.time()

        def admissible_h(assignment, used):
            depth = len(assignment)
            if depth == n_active:
                return 0.0
            unused = [p for p in range(N) if p not in used]
            h = 0.0
            for i in range(depth, n_active):
                li = L[i]
                for (lj, w) in logical_adj[li]:
                    j = L_index[lj]
                    if j < depth:
                        p_j = assignment[j]
                        if unused:
                            md = min(dist[p_j][u] for u in unused)
                        else:
                            md = 0
                        h += w * md
                    elif j > i:
                        h += w * min_edge_dist
            return h

        counter = 0
        heap = [(0.0, 0, 0.0, (), frozenset())]
        best_g = float('inf')
        nodes = 0

        while heap:
            if nodes >= node_budget or (time.time() - start_t) > time_budget:
                break
            f, _, g, assignment, used = heapq.heappop(heap)
            if f >= best_g:
                break
            depth = len(assignment)
            if depth == n_active:
                if g < best_g:
                    best_g = g
                    best_assignment = assignment
                continue
            nodes += 1
            li = L[depth]

            placed_neigh = []
            for (lj, w) in logical_adj[li]:
                j = L_index[lj]
                if j < depth:
                    placed_neigh.append((assignment[j], w))

            def score(p):
                if placed_neigh:
                    return sum(w * dist[p][pj] for pj, w in placed_neigh)
                return -centrality.get(p, 0.0)

            free = [p for p in range(N) if p not in used]
            free.sort(key=score)
            cands = free[:branching]

            for p in cands:
                delta = 0.0
                for (pj, w) in placed_neigh:
                    delta += w * dist[p][pj]
                new_g = g + delta
                if new_g >= best_g:
                    continue
                new_assign = assignment + (p,)
                new_used = used | {p}
                new_h = admissible_h(new_assign, new_used)
                new_f = new_g + new_h
                if new_f >= best_g:
                    continue
                counter += 1
                heapq.heappush(heap, (new_f, counter, new_g, new_assign, new_used))

        if best_assignment is not None:
            for i, p in enumerate(best_assignment):
                mapping[L[i]] = p
                used_phys.add(p)
        else:
            for i, li in enumerate(L):
                placed_neigh = []
                for (lj, w) in logical_adj[li]:
                    j = L_index[lj]
                    if j < i and mapping[L[j]] != -1:
                        placed_neigh.append((mapping[L[j]], w))
                best_p = None
                best_s = float('inf')
                for p in range(N):
                    if p in used_phys:
                        continue
                    if placed_neigh:
                        s = sum(w * dist[p][pj] for pj, w in placed_neigh)
                    else:
                        s = -centrality.get(p, 0.0)
                    if s < best_s:
                        best_s = s
                        best_p = p
                if best_p is None:
                    break
                mapping[li] = best_p
                used_phys.add(best_p)

    remaining_phys = [p for p in physicals_sorted if p not in used_phys]
    rp_idx = 0
    for l in range(N):
        if mapping[l] == -1:
            if rp_idx < len(remaining_phys):
                mapping[l] = remaining_phys[rp_idx]
                used_phys.add(remaining_phys[rp_idx])
                rp_idx += 1

    seen = set()
    for i in range(N):
        p = mapping[i]
        if p < 0 or p >= N or p in seen:
            mapping[i] = -1
        else:
            seen.add(p)
    avail = [p for p in range(N) if p not in seen]
    ai = 0
    for i in range(N):
        if mapping[i] == -1:
            mapping[i] = avail[ai]
            ai += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = [0] * N
    for l, p in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)