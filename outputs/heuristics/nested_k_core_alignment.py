def init_mapping(self):
    from collections import defaultdict, deque

    N = self.num_qubits

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    def k_core_decomposition(adj, nodes):
        deg = {v: len(adj[v]) for v in nodes}
        coreness = {v: 0 for v in nodes}
        remaining = set(nodes)
        cur_adj = {v: set(adj[v]) & remaining for v in nodes}
        k = 0
        while remaining:
            changed = True
            while changed:
                changed = False
                to_remove = [v for v in remaining if deg[v] <= k]
                if to_remove:
                    changed = True
                    for v in to_remove:
                        coreness[v] = k
                        remaining.discard(v)
                        for u in list(cur_adj[v]):
                            cur_adj[u].discard(v)
                            deg[u] -= 1
                        cur_adj[v].clear()
            k += 1
            if not remaining:
                break
        return coreness

    logical_adj = defaultdict(set)
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            if q1 == q2:
                continue
            logical_adj[q1].add(q2)
            logical_adj[q2].add(q1)
            logical_nodes.add(q1)
            logical_nodes.add(q2)

    physical_nodes = set(range(N))
    physical_adj = defaultdict(set)
    for (a, b) in self.backend_connections:
        if a == b:
            continue
        physical_adj[a].add(b)
        physical_adj[b].add(a)
    for p in physical_nodes:
        physical_adj.setdefault(p, set())

    if logical_nodes:
        kL = k_core_decomposition(logical_adj, logical_nodes)
    else:
        kL = {}
    kP = k_core_decomposition(physical_adj, physical_nodes)

    logical_activity = self.logical_activity if self.logical_activity else {}
    physical_centrality = self.physical_centrality if self.physical_centrality else {}

    logical_shells = defaultdict(list)
    for q, c in kL.items():
        logical_shells[c].append(q)
    physical_shells = defaultdict(list)
    for p, c in kP.items():
        physical_shells[c].append(p)

    for c in logical_shells:
        logical_shells[c].sort(key=lambda q: -logical_activity.get(q, 0))
    for c in physical_shells:
        physical_shells[c].sort(key=lambda p: -physical_centrality.get(p, 0))

    used_physical = set()
    placed_logical = set()

    physical_pool_by_core = sorted(physical_shells.keys(), reverse=True)
    pool_pointers = {c: 0 for c in physical_pool_by_core}

    def take_best_physical(prefer_near=None):
        if prefer_near is not None and prefer_near:
            best_p = -1
            best_d = float('inf')
            best_c = -1
            for c in physical_pool_by_core:
                for p in physical_shells[c]:
                    if p in used_physical:
                        continue
                    d = 0
                    for anchor in prefer_near:
                        if anchor < len(self.distance_matrix) and p < len(self.distance_matrix[anchor]):
                            d += self.distance_matrix[anchor][p]
                        else:
                            d += 0
                    cent = physical_centrality.get(p, 0)
                    if (d < best_d) or (d == best_d and (c > best_c or (c == best_c and cent > physical_centrality.get(best_p, -1)))):
                        best_d = d
                        best_p = p
                        best_c = c
                if best_p != -1:
                    pass
            if best_p != -1:
                used_physical.add(best_p)
                return best_p
        for c in physical_pool_by_core:
            for p in physical_shells[c]:
                if p not in used_physical:
                    used_physical.add(p)
                    return p
        return -1

    for c in sorted(logical_shells.keys(), reverse=True):
        shell_qubits = logical_shells[c]
        for q in shell_qubits:
            if q >= N:
                continue
            placed_neighbors = [self.mapping_dict[nb] for nb in logical_adj[q]
                                if nb in placed_logical and self.mapping_dict[nb] != -1]
            phys = take_best_physical(prefer_near=placed_neighbors)
            if phys == -1:
                continue
            self.mapping_dict[q] = phys
            self.reverse_mapping_dict[phys] = q
            placed_logical.add(q)

    remaining_logical = [q for q in range(N) if self.mapping_dict[q] == -1]
    remaining_physical = [p for p in range(N) if p not in used_physical]

    for q in remaining_logical:
        if q < N and q not in used_physical and self.mapping_dict[q] == -1:
            if q in remaining_physical:
                self.mapping_dict[q] = q
                self.reverse_mapping_dict[q] = q
                used_physical.add(q)
                remaining_physical.remove(q)

    remaining_logical = [q for q in range(N) if self.mapping_dict[q] == -1]
    remaining_physical = [p for p in range(N) if p not in used_physical]
    for q, p in zip(remaining_logical, remaining_physical):
        self.mapping_dict[q] = p
        self.reverse_mapping_dict[p] = q
        used_physical.add(p)

    for q in range(N):
        if self.mapping_dict[q] == -1:
            for p in range(N):
                if p not in used_physical:
                    self.mapping_dict[q] = p
                    self.reverse_mapping_dict[p] = q
                    used_physical.add(p)
                    break

    if self.use_isl:
        try:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        except Exception:
            self.isl_mapping = None

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)