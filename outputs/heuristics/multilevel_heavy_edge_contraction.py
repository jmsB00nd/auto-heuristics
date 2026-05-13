def init_mapping(self):
    import heapq
    from collections import defaultdict

    N = self.num_qubits

    logical_qubits = set()
    edge_weight = defaultdict(float)
    if getattr(self, "qubit_interaction_graph", None):
        for u, nbrs in self.qubit_interaction_graph.items():
            logical_qubits.add(u)
            for v, w in nbrs.items():
                logical_qubits.add(v)
                if u < v and w > 0:
                    edge_weight[(u, v)] += float(w)
    if not edge_weight:
        for _gid, qs in self.access.items():
            if len(qs) == 2:
                a, b = qs
                if a == b:
                    continue
                logical_qubits.add(a); logical_qubits.add(b)
                key = (a, b) if a < b else (b, a)
                edge_weight[key] += 1.0
    for _gid, qs in self.access.items():
        for q in qs:
            logical_qubits.add(q)

    logical_qubits = {q for q in logical_qubits if 0 <= q < N}

    parent_of_child = {}
    merge_stack = []
    super_nodes = set(logical_qubits)
    super_adj = defaultdict(lambda: defaultdict(float))
    for (u, v), w in edge_weight.items():
        if u in super_nodes and v in super_nodes:
            super_adj[u][v] += w
            super_adj[v][u] += w

    next_id = (max(super_nodes) + 1) if super_nodes else N
    target_size = min(len(super_nodes), N)

    while len(super_nodes) > 1:
        heap = []
        for u in super_nodes:
            for v, w in super_adj[u].items():
                if u < v and w > 0:
                    heapq.heappush(heap, (-w, u, v))
        if not heap:
            break
        matched = set()
        contractions = []
        while heap:
            neg_w, u, v = heapq.heappop(heap)
            if u in matched or v in matched:
                continue
            if u not in super_nodes or v not in super_nodes:
                continue
            matched.add(u); matched.add(v)
            contractions.append((u, v))
        if not contractions:
            break
        for (u, v) in contractions:
            s = next_id; next_id += 1
            merge_stack.append((s, u, v))
            parent_of_child[u] = s
            parent_of_child[v] = s
            new_nbrs = defaultdict(float)
            for x, w in super_adj[u].items():
                if x != v:
                    new_nbrs[x] += w
            for x, w in super_adj[v].items():
                if x != u:
                    new_nbrs[x] += w
            for x in list(super_adj[u].keys()):
                if x in super_adj and u in super_adj[x]:
                    del super_adj[x][u]
            for x in list(super_adj[v].keys()):
                if x in super_adj and v in super_adj[x]:
                    del super_adj[x][v]
            del super_adj[u]; del super_adj[v]
            super_nodes.discard(u); super_nodes.discard(v)
            super_nodes.add(s)
            for x, w in new_nbrs.items():
                super_adj[s][x] += w
                super_adj[x][s] += w
        if len(super_nodes) <= 2:
            break

    centrality = getattr(self, "physical_centrality", {}) or {}
    phys_order = sorted(range(N), key=lambda p: -centrality.get(p, 0.0))

    used_phys = set()
    super_to_phys = {}

    super_weight = {s: sum(super_adj[s].values()) for s in super_nodes}
    remaining = sorted(super_nodes, key=lambda s: -super_weight.get(s, 0.0))

    def pick_anchor():
        for p in phys_order:
            if p not in used_phys:
                return p
        return None

    def pick_near(placed_phys_set):
        best = None; best_score = None
        for p in range(N):
            if p in used_phys:
                continue
            score = 0.0
            for pp in placed_phys_set:
                d = self.distance_matrix[p][pp] if pp < len(self.distance_matrix) and p < len(self.distance_matrix[pp]) else N
                if d <= 0:
                    d = N
                score += 1.0 / d
            if best is None or score > best_score:
                best = p; best_score = score
        return best if best is not None else pick_anchor()

    placed_phys = set()
    if remaining:
        first = remaining[0]
        anchor = pick_anchor()
        if anchor is None:
            anchor = 0
        super_to_phys[first] = anchor
        used_phys.add(anchor); placed_phys.add(anchor)
        for s in remaining[1:]:
            neighbors_placed = [super_to_phys[t] for t in super_adj[s] if t in super_to_phys]
            if neighbors_placed:
                cand = None; cand_score = None
                for p in range(N):
                    if p in used_phys:
                        continue
                    score = 0.0
                    for pp in neighbors_placed:
                        d = self.distance_matrix[p][pp]
                        if d <= 0:
                            d = N
                        score += super_adj[s].get(self.reverse_super(pp) if False else 0, 0.0) / d if False else 1.0 / d
                    if cand is None or score > cand_score:
                        cand = p; cand_score = score
                p = cand if cand is not None else pick_anchor()
            else:
                p = pick_near(placed_phys) if placed_phys else pick_anchor()
            if p is None:
                p = pick_anchor()
            super_to_phys[s] = p
            used_phys.add(p); placed_phys.add(p)

    placed_log = {}
    for s, p in super_to_phys.items():
        placed_log[s] = p

    while merge_stack:
        s, a, b = merge_stack.pop()
        if s not in placed_log:
            continue
        p = placed_log.pop(s)
        placed_log[a] = p
        nbrs = self.backend.get(p, set()) if hasattr(self.backend, "get") else self.backend[p]
        free_neighbors = [q for q in nbrs if q not in used_phys]
        if free_neighbors:
            best_q = None; best_d = None
            for q in free_neighbors:
                d = self.distance_matrix[p][q]
                if best_q is None or d < best_d:
                    best_q = q; best_d = d
            q_pick = best_q
        else:
            q_pick = None
            best_d = None
            for q in range(N):
                if q in used_phys or q == p:
                    continue
                d = self.distance_matrix[p][q]
                if d <= 0:
                    continue
                if q_pick is None or d < best_d:
                    q_pick = q; best_d = d
            if q_pick is None:
                for q in range(N):
                    if q not in used_phys:
                        q_pick = q; break
        placed_log[b] = q_pick
        used_phys.add(q_pick)

    final_log_to_phys = {}
    for L, p in placed_log.items():
        if 0 <= L < N and p is not None and 0 <= p < N:
            final_log_to_phys[L] = p

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N
    used_phys2 = set()
    for L, p in final_log_to_phys.items():
        if self.mapping_dict[L] == -1 and p not in used_phys2:
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L
            used_phys2.add(p)

    free_phys_central = [p for p in phys_order if p not in used_phys2]
    unplaced_logicals = [L for L in range(N) if self.mapping_dict[L] == -1]
    interacting_unplaced = [L for L in unplaced_logicals if L in logical_qubits]
    idle_unplaced = [L for L in unplaced_logicals if L not in logical_qubits]

    fp_iter = iter(free_phys_central)
    for L in interacting_unplaced:
        try:
            p = next(fp_iter)
        except StopIteration:
            break
        self.mapping_dict[L] = p
        self.reverse_mapping_dict[p] = L
        used_phys2.add(p)

    free_phys_remaining = [p for p in range(N) if p not in used_phys2]
    fp_iter2 = iter(free_phys_remaining)
    for L in idle_unplaced:
        try:
            p = next(fp_iter2)
        except StopIteration:
            break
        self.mapping_dict[L] = p
        self.reverse_mapping_dict[p] = L
        used_phys2.add(p)

    free_phys_final = [p for p in range(N) if p not in used_phys2]
    fp_iter3 = iter(free_phys_final)
    for L in range(N):
        if self.mapping_dict[L] == -1:
            try:
                p = next(fp_iter3)
            except StopIteration:
                for q in range(N):
                    if q not in used_phys2:
                        p = q; break
                else:
                    p = L
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L
            used_phys2.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)