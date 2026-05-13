def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    def k_core_layers(adj, nodes):
        deg = {v: len(adj[v]) for v in nodes}
        remaining = set(nodes)
        coreness = {}
        while remaining:
            min_d = min(deg[v] for v in remaining)
            to_peel = [v for v in remaining if deg[v] <= min_d]
            while to_peel:
                v = to_peel.pop()
                if v not in remaining:
                    continue
                coreness[v] = min_d
                remaining.discard(v)
                for u in adj[v]:
                    if u in remaining:
                        deg[u] -= 1
                        if deg[u] <= min_d:
                            to_peel.append(u)
        layers = defaultdict(list)
        for v, c in coreness.items():
            layers[c].append(v)
        return layers

    logical_adj = defaultdict(set)
    logical_nodes = set()
    edge_weight = defaultdict(int)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_adj[a].add(b)
            logical_adj[b].add(a)
            logical_nodes.add(a)
            logical_nodes.add(b)
            key = (a, b) if a < b else (b, a)
            edge_weight[key] += 1

    for q in range(N):
        logical_nodes.add(q)
        _ = logical_adj[q]

    physical_adj = defaultdict(set)
    for p in range(N):
        for nb in self.backend.get(p, ()):
            if 0 <= nb < N and nb != p:
                physical_adj[p].add(nb)
                physical_adj[nb].add(p)
    physical_nodes = set(range(N))

    log_layers = k_core_layers(logical_adj, logical_nodes)
    phys_layers = k_core_layers(physical_adj, physical_nodes)

    log_cores_desc = sorted(log_layers.keys(), reverse=True)
    phys_cores_desc = sorted(phys_layers.keys(), reverse=True)

    activity = getattr(self, "logical_activity", {}) or {}
    centrality = getattr(self, "physical_centrality", {}) or {}

    def log_sort_key(q):
        return (-activity.get(q, 0), -len(logical_adj[q]), q)

    def phys_sort_key(p):
        return (-centrality.get(p, 0.0), -len(physical_adj[p]), p)

    mapping = [-1] * N
    used_phys = set()
    used_log = set()

    phys_pool = []
    phys_idx = 0

    def refill_phys_pool(up_to_core_index):
        nonlocal phys_pool
        while phys_idx_holder[0] <= up_to_core_index and phys_idx_holder[0] < len(phys_cores_desc):
            c = phys_cores_desc[phys_idx_holder[0]]
            phys_pool.extend(sorted(phys_layers[c], key=phys_sort_key))
            phys_idx_holder[0] += 1

    phys_idx_holder = [0]

    for li, lc in enumerate(log_cores_desc):
        logicals = sorted(log_layers[lc], key=log_sort_key)
        if li < len(phys_cores_desc):
            refill_phys_pool(li)
        else:
            refill_phys_pool(len(phys_cores_desc) - 1)

        for lq in logicals:
            if lq in used_log:
                continue
            chosen = -1
            while phys_pool:
                cand = phys_pool.pop(0)
                if cand not in used_phys:
                    chosen = cand
                    break
            if chosen == -1:
                next_idx = phys_idx_holder[0]
                while chosen == -1 and next_idx < len(phys_cores_desc):
                    phys_pool.extend(sorted(phys_layers[phys_cores_desc[next_idx]], key=phys_sort_key))
                    next_idx += 1
                    phys_idx_holder[0] = next_idx
                    while phys_pool:
                        cand = phys_pool.pop(0)
                        if cand not in used_phys:
                            chosen = cand
                            break
            if chosen == -1:
                break
            mapping[lq] = chosen
            used_phys.add(chosen)
            used_log.add(lq)

    remaining_logicals = [q for q in range(N) if q not in used_log]
    remaining_logicals.sort(key=log_sort_key)
    remaining_physicals = [p for p in range(N) if p not in used_phys]
    remaining_physicals.sort(key=phys_sort_key)

    for lq in remaining_logicals:
        if not remaining_physicals:
            break
        pq = remaining_physicals.pop(0)
        mapping[lq] = pq
        used_phys.add(pq)
        used_log.add(lq)

    if any(m == -1 for m in mapping):
        leftover_phys = [p for p in range(N) if p not in used_phys]
        for lq in range(N):
            if mapping[lq] == -1:
                if leftover_phys:
                    pq = leftover_phys.pop(0)
                else:
                    for p in range(N):
                        if p not in used_phys:
                            pq = p
                            break
                    else:
                        pq = lq
                mapping[lq] = pq
                used_phys.add(pq)

    if len(set(mapping)) != N:
        seen = set()
        free = [p for p in range(N) if p not in set(mapping)]
        for i in range(N):
            if mapping[i] in seen:
                mapping[i] = free.pop() if free else i
            seen.add(mapping[i])

    self.mapping_dict = list(mapping)
    self.reverse_mapping_dict = [0] * N
    for lq, pq in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)