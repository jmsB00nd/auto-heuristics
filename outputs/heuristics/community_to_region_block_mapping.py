def init_mapping(self):
    import random
    from collections import defaultdict, deque

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # ---- 1. Collect logical qubits & weighted logical graph ----
    logical_qubits = set()
    for _gid, qs in self.access.items():
        for q in qs:
            if 0 <= q < N:
                logical_qubits.add(q)

    log_adj = defaultdict(lambda: defaultdict(float))
    for u in logical_qubits:
        row = self.qubit_interaction_graph.get(u, {}) if hasattr(self.qubit_interaction_graph, "get") else self.qubit_interaction_graph[u]
        for v, w in row.items():
            if v in logical_qubits and v != u and w > 0:
                log_adj[u][v] = float(w)
                log_adj[v][u] = float(w)

    # ---- 2. Weighted label propagation ----
    rng = random.Random(0xC0DE)
    labels = {q: q for q in logical_qubits}
    nodes = list(logical_qubits)
    max_iter = 20
    for _ in range(max_iter):
        rng.shuffle(nodes)
        changed = False
        for u in nodes:
            nbrs = log_adj.get(u)
            if not nbrs:
                continue
            score = defaultdict(float)
            for v, w in nbrs.items():
                score[labels[v]] += w
            best_label = labels[u]
            best_score = score.get(best_label, -1.0)
            for lab, s in score.items():
                if s > best_score or (s == best_score and lab < best_label):
                    best_score = s
                    best_label = lab
            if best_label != labels[u]:
                labels[u] = best_label
                changed = True
        if not changed:
            break

    communities = defaultdict(list)
    for q, lab in labels.items():
        communities[lab].append(q)
    community_list = list(communities.values())

    # community internal weight (for tiebreak / matching)
    def _internal_weight(members):
        mset = set(members)
        tot = 0.0
        for u in members:
            for v, w in log_adj.get(u, {}).items():
                if v in mset and v > u:
                    tot += w
        return tot

    community_list.sort(key=lambda c: (-len(c), -_internal_weight(c)))

    # ---- 3. Pick well-separated high-centrality seeds ----
    centrality = self.physical_centrality if isinstance(self.physical_centrality, dict) else {i: self.physical_centrality[i] for i in range(N)}
    phys_by_cent = sorted(range(N), key=lambda p: (-centrality.get(p, 0.0), p))

    num_comms = len(community_list)
    # adaptive separation: prefer larger spacing, relax if not enough seeds found
    seeds = []
    if num_comms > 0:
        # try several thresholds, from large to small
        max_d = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = self.distance_matrix[i][j]
                if d > max_d:
                    max_d = d
        thresholds = []
        if max_d >= 1:
            for t in (max_d, max_d // 2, max_d // 3, 2, 1, 0):
                if t not in thresholds:
                    thresholds.append(t)
        else:
            thresholds = [0]
        for thr in thresholds:
            seeds = []
            for p in phys_by_cent:
                ok = True
                for s in seeds:
                    if self.distance_matrix[p][s] < thr:
                        ok = False
                        break
                if ok:
                    seeds.append(p)
                    if len(seeds) == num_comms:
                        break
            if len(seeds) == num_comms:
                break

    # ---- 4. Grow BFS-ball regions, claim physicals ----
    used_phys = [False] * N
    regions = []  # list of lists of physical qubits, aligned with community_list
    # Order communities by size desc (already), grow regions in same order.
    for idx, members in enumerate(community_list):
        target = len(members)
        if idx >= len(seeds):
            regions.append([])
            continue
        seed = seeds[idx]
        region = []
        if not used_phys[seed]:
            used_phys[seed] = True
            region.append(seed)
        # BFS expansion
        visited = {seed}
        frontier = deque([seed])
        while frontier and len(region) < target:
            u = frontier.popleft()
            nbrs = sorted(self.backend.get(u, ()), key=lambda x: (-centrality.get(x, 0.0), self.distance_matrix[seed][x], x))
            for v in nbrs:
                if v in visited:
                    continue
                visited.add(v)
                frontier.append(v)
                if not used_phys[v]:
                    used_phys[v] = True
                    region.append(v)
                    if len(region) >= target:
                        break
        regions.append(region)

    # ---- 5. Lay out each community inside its region ----
    def _intra_degree(u, mset):
        s = 0.0
        for v, w in log_adj.get(u, {}).items():
            if v in mset:
                s += w
        return s

    assigned_logical = set()
    for members, region in zip(community_list, regions):
        if not region:
            continue
        mset = set(members)
        ordered_logicals = sorted(members, key=lambda q: (-_intra_degree(q, mset), q))
        ordered_phys = sorted(region, key=lambda p: (-centrality.get(p, 0.0), p))
        for L, P in zip(ordered_logicals, ordered_phys):
            self.mapping_dict[L] = P
            self.reverse_mapping_dict[P] = L
            assigned_logical.add(L)

    # ---- 6. Fallback: place any unassigned logicals on free physicals ----
    free_phys = [p for p in range(N) if not used_phys[p]]
    free_set = set(free_phys)
    free_sorted = sorted(free_phys, key=lambda p: (-centrality.get(p, 0.0), p))

    for L in range(N):
        if self.mapping_dict[L] != -1:
            continue
        # prefer identity if free
        if L in free_set:
            P = L
            free_set.remove(P)
            free_sorted.remove(P)
        elif free_sorted:
            P = free_sorted.pop(0)
            free_set.discard(P)
        else:
            # extremely defensive: pick any still-unused
            P = next((p for p in range(N) if not used_phys[p] and p not in set(self.mapping_dict)), -1)
            if P == -1:
                # find any physical not in mapping_dict
                taken = set(self.mapping_dict)
                P = next(p for p in range(N) if p not in taken)
        used_phys[P] = True
        self.mapping_dict[L] = P
        self.reverse_mapping_dict[P] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)