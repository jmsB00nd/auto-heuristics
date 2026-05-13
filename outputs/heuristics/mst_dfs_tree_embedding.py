def init_mapping(self):
    import collections

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # ---- 1. Collect logical interactions ----
    logical_qubits = set()
    edge_weights = {}
    for _, qubits in self.access.items():
        for q in qubits:
            logical_qubits.add(q)
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            try:
                w = self.qubit_interaction_graph[a][b]
            except Exception:
                w = 0
            if not w or w <= 0:
                w = edge_weights.get(key, 0) + 1
            if w > edge_weights.get(key, 0):
                edge_weights[key] = w

    phys_centrality = getattr(self, "physical_centrality", {}) or {}

    # Helper: BFS on backend to find nearest unused physical from a starting node.
    def nearest_unused(start, used):
        if start not in used:
            return start
        seen = {start}
        dq = collections.deque([start])
        while dq:
            cur = dq.popleft()
            for nb in self.backend[cur]:
                if nb in seen:
                    continue
                seen.add(nb)
                if nb not in used:
                    return nb
                dq.append(nb)
        return None

    # ---- 2. Maximum-Weight Spanning Tree via Kruskal ----
    mst_adj = collections.defaultdict(list)
    if logical_qubits and edge_weights:
        parent = {q: q for q in logical_qubits}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        for (u, v), w in sorted(edge_weights.items(), key=lambda kv: -kv[1]):
            if union(u, v):
                mst_adj[u].append((v, w))
                mst_adj[v].append((u, w))

    activity = {q: sum(w for _, w in mst_adj[q]) for q in logical_qubits}

    # ---- 3. BFS spanning tree of the coupling graph ----
    phys_root = max(range(N), key=lambda p: phys_centrality.get(p, 0)) if N > 0 else 0
    bfs_children = collections.defaultdict(list)
    if N > 0:
        visited_phys = {phys_root}
        dq = collections.deque([phys_root])
        while dq:
            u = dq.popleft()
            nbrs = sorted(self.backend[u], key=lambda p: -phys_centrality.get(p, 0))
            for v in nbrs:
                if v not in visited_phys:
                    visited_phys.add(v)
                    bfs_children[u].append(v)
                    dq.append(v)

    used_phys = set()

    # ---- 4. Synchronized DFS embedding (iterative) ----
    def embed_iter(root_log, root_phys):
        stack = [(root_log, root_phys, None)]
        while stack:
            log_node, phys_hint, log_parent = stack.pop()
            if self.mapping_dict[log_node] != -1:
                continue
            phys_node = nearest_unused(phys_hint, used_phys)
            if phys_node is None:
                continue
            self.mapping_dict[log_node] = phys_node
            self.reverse_mapping_dict[phys_node] = log_node
            used_phys.add(phys_node)

            log_children = [(c, w) for c, w in mst_adj[log_node]
                            if c != log_parent and self.mapping_dict[c] == -1]
            log_children.sort(key=lambda cw: -cw[1])

            avail = [p for p in bfs_children[phys_node] if p not in used_phys]
            if len(avail) < len(log_children):
                extras = [p for p in self.backend[phys_node]
                          if p not in used_phys and p not in avail]
                avail = avail + extras
            avail.sort(key=lambda p: -phys_centrality.get(p, 0))

            pairs = list(zip(log_children, avail))
            unmatched = log_children[len(pairs):]
            for (c_log, _), c_phys in reversed(pairs):
                stack.append((c_log, c_phys, log_node))
            for (c_log, _) in reversed(unmatched):
                stack.append((c_log, phys_node, log_node))

    # ---- 5. Iterate MST connected components ordered by activity ----
    visited_log = set()
    component_roots = []
    for q in sorted(logical_qubits, key=lambda x: -activity.get(x, 0)):
        if q in visited_log:
            continue
        component_roots.append(q)
        dq = collections.deque([q])
        visited_log.add(q)
        while dq:
            u = dq.popleft()
            for v, _ in mst_adj[u]:
                if v not in visited_log:
                    visited_log.add(v)
                    dq.append(v)

    for c_root in component_roots:
        if len(used_phys) >= N:
            break
        if not used_phys:
            phys_start = phys_root
        else:
            free = [p for p in range(N) if p not in used_phys]
            if not free:
                break
            phys_start = max(free, key=lambda p: phys_centrality.get(p, 0))
        embed_iter(c_root, phys_start)

    # ---- 6. Back-fill remaining logicals onto unused physicals ----
    unused = sorted((p for p in range(N) if p not in used_phys),
                    key=lambda p: -phys_centrality.get(p, 0))
    for L in range(N):
        if self.mapping_dict[L] == -1:
            if not unused:
                break
            p = unused.pop(0)
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L
            used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)