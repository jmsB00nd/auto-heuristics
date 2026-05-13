def init_mapping(self):
    from collections import defaultdict, deque

    N = self.num_qubits

    logical_adj = defaultdict(lambda: defaultdict(int))
    logical_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_adj[a][b] += 1
            logical_adj[b][a] += 1
            logical_set.add(a)
            logical_set.add(b)

    logical_degree = {q: len(logical_adj[q]) for q in logical_set}
    logical_weight = {q: sum(logical_adj[q].values()) for q in logical_set}

    physical_degree = [len(self.backend[p]) for p in range(N)]
    centrality = self.physical_centrality if isinstance(self.physical_centrality, dict) else {p: 0.0 for p in range(N)}

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N
    used_phys = set()
    placed_logical = {}

    def pick_seed_logical():
        if not logical_set:
            return None
        best = None
        best_key = None
        for q in logical_set:
            key = (logical_weight.get(q, 0), self.logical_activity.get(q, 0), logical_degree.get(q, 0), -q)
            if best_key is None or key > best_key:
                best_key = key
                best = q
        return best

    def pick_seed_physical():
        best = None
        best_key = None
        for p in range(N):
            key = (centrality.get(p, 0.0), physical_degree[p], -p)
            if best_key is None or key > best_key:
                best_key = key
                best = p
        return best if best is not None else 0

    def bfs_order(seed):
        order = []
        visited = {seed}
        dq = deque([seed])
        while dq:
            u = dq.popleft()
            order.append(u)
            neighbors = sorted(
                logical_adj[u].items(),
                key=lambda kv: (-kv[1], -logical_weight.get(kv[0], 0), kv[0]),
            )
            for v, _w in neighbors:
                if v not in visited:
                    visited.add(v)
                    dq.append(v)
        return order, visited

    def score_candidate(logical_q, phys_q):
        adj_score = 0
        weighted_adj = 0
        for nb_log, w in logical_adj[logical_q].items():
            if nb_log in placed_logical:
                nb_phys = placed_logical[nb_log]
                if nb_phys in self.backend[phys_q]:
                    adj_score += 1
                    weighted_adj += w
        ldeg = logical_degree.get(logical_q, 0)
        pdeg = physical_degree[phys_q]
        deg_compat = -abs(pdeg - ldeg) if pdeg >= ldeg else -(ldeg - pdeg) * 2
        cent = centrality.get(phys_q, 0.0)
        return (adj_score, weighted_adj, deg_compat, cent, -phys_q)

    def place(logical_q):
        best_phys = None
        best_key = None
        for p in range(N):
            if p in used_phys:
                continue
            key = score_candidate(logical_q, p)
            if best_key is None or key > best_key:
                best_key = key
                best_phys = p
        if best_phys is None:
            return False
        self.mapping_dict[logical_q] = best_phys
        self.reverse_mapping_dict[best_phys] = logical_q
        used_phys.add(best_phys)
        placed_logical[logical_q] = best_phys
        return True

    seed_logical = pick_seed_logical()
    if seed_logical is not None:
        seed_phys = pick_seed_physical()
        self.mapping_dict[seed_logical] = seed_phys
        self.reverse_mapping_dict[seed_phys] = seed_logical
        used_phys.add(seed_phys)
        placed_logical[seed_logical] = seed_phys

        order, visited = bfs_order(seed_logical)
        for lq in order:
            if lq == seed_logical:
                continue
            place(lq)

        remaining_logical = [q for q in logical_set if q not in visited]
        remaining_logical.sort(key=lambda q: (-logical_weight.get(q, 0), -logical_degree.get(q, 0), q))
        for lq in remaining_logical:
            place(lq)

    remaining_logicals_all = [l for l in range(N) if self.mapping_dict[l] == -1]
    remaining_phys_sorted = sorted(
        [p for p in range(N) if p not in used_phys],
        key=lambda p: (-centrality.get(p, 0.0), -physical_degree[p], p),
    )

    for lq in remaining_logicals_all:
        if lq < N and lq not in used_phys and self.reverse_mapping_dict[lq] == -1:
            self.mapping_dict[lq] = lq
            self.reverse_mapping_dict[lq] = lq
            used_phys.add(lq)

    remaining_logicals_all = [l for l in range(N) if self.mapping_dict[l] == -1]
    remaining_phys_sorted = [p for p in range(N) if p not in used_phys]
    remaining_phys_sorted.sort(key=lambda p: (-centrality.get(p, 0.0), -physical_degree[p], p))

    for lq, p in zip(remaining_logicals_all, remaining_phys_sorted):
        self.mapping_dict[lq] = p
        self.reverse_mapping_dict[p] = lq
        used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)