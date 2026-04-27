def init_mapping(self):
    from collections import defaultdict, deque

    N = self.num_qubits

    edge_weight = defaultdict(int)
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                logical_nodes.add(a)
                continue
            u, v = (a, b) if a < b else (b, a)
            edge_weight[(u, v)] += 1
            logical_nodes.add(a)
            logical_nodes.add(b)
        elif len(qubits) == 1:
            logical_nodes.add(qubits[0])

    for l in range(N):
        logical_nodes.add(l)

    parent = {l: l for l in logical_nodes}
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

    edges_sorted = sorted(edge_weight.items(), key=lambda kv: -kv[1])
    mst_adj = defaultdict(list)
    mst_degree = defaultdict(int)
    for (u, v), w in edges_sorted:
        if union(u, v):
            mst_adj[u].append((v, w))
            mst_adj[v].append((u, w))
            mst_degree[u] += 1
            mst_degree[v] += 1

    active = [l for l in logical_nodes if l in mst_degree or any(l in qs for qs in self.access.values())]
    if not active:
        active = sorted(logical_nodes)

    weighted_logical_deg = defaultdict(int)
    for (u, v), w in edge_weight.items():
        weighted_logical_deg[u] += w
        weighted_logical_deg[v] += w

    logical_root = max(active, key=lambda l: (weighted_logical_deg[l], mst_degree[l], -l))

    logical_order = []
    seen_log = set()
    q = deque([logical_root])
    seen_log.add(logical_root)
    while q:
        node = q.popleft()
        logical_order.append(node)
        neighbors = sorted(mst_adj[node], key=lambda nw: -nw[1])
        for nb, _w in neighbors:
            if nb not in seen_log:
                seen_log.add(nb)
                q.append(nb)
    for l in sorted(logical_nodes):
        if l not in seen_log:
            seen_log.add(l)
            logical_order.append(l)

    phys_degree = [0] * N
    for p in range(N):
        nbrs = self.backend[p] if p < len(self.backend) else []
        phys_degree[p] = len(nbrs)

    def total_dist(p):
        try:
            row = self.distance_matrix[p]
            return sum(d for d in row if d > 0)
        except Exception:
            return 0

    hub = max(range(N), key=lambda p: (phys_degree[p], -total_dist(p), -p))

    phys_order = []
    seen_phys = set()
    q = deque([hub])
    seen_phys.add(hub)
    while q:
        node = q.popleft()
        phys_order.append(node)
        nbrs = self.backend[node] if node < len(self.backend) else []
        nbrs_sorted = sorted(nbrs, key=lambda x: (-phys_degree[x], x))
        for nb in nbrs_sorted:
            if nb not in seen_phys:
                seen_phys.add(nb)
                q.append(nb)
    for p in range(N):
        if p not in seen_phys:
            seen_phys.add(p)
            phys_order.append(p)

    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = [False] * N

    phys_iter = iter(phys_order)
    for l in logical_order:
        if mapping[l] != -1:
            continue
        assigned = False
        for p in phys_iter:
            if not used_phys[p]:
                mapping[l] = p
                reverse[p] = l
                used_phys[p] = True
                assigned = True
                break
        if not assigned:
            for p in range(N):
                if not used_phys[p]:
                    mapping[l] = p
                    reverse[p] = l
                    used_phys[p] = True
                    break

    for l in range(N):
        if mapping[l] == -1:
            for p in range(N):
                if not used_phys[p]:
                    mapping[l] = p
                    reverse[p] = l
                    used_phys[p] = True
                    break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)