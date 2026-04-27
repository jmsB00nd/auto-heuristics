def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    # --- 1. Logical interaction graph (weighted) ---
    logical_neighbors = defaultdict(lambda: defaultdict(float))
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_neighbors[a][b] += 1.0
            logical_neighbors[b][a] += 1.0
            logical_qubits.add(a)
            logical_qubits.add(b)

    # --- 2. Physical graph (unweighted) ---
    physical_neighbors = defaultdict(set)
    for (u, v) in self.backend_connections:
        if u == v:
            continue
        if 0 <= u < N and 0 <= v < N:
            physical_neighbors[u].add(v)
            physical_neighbors[v].add(u)

    # --- 3a. Weighted local clustering coefficient (logical) ---
    def weighted_clustering(node, nbr_dict):
        nbrs = list(nbr_dict[node].keys())
        k = len(nbrs)
        if k < 2:
            return 0.0
        # max edge weight for normalization
        max_w = 0.0
        for x in nbr_dict:
            for w in nbr_dict[x].values():
                if w > max_w:
                    max_w = w
        if max_w <= 0:
            return 0.0
        triangle_strength = 0.0
        nbr_set = set(nbrs)
        for i in range(k):
            for j in range(i + 1, k):
                u, v = nbrs[i], nbrs[j]
                if v in nbr_dict[u]:
                    w_uv_node = (nbr_dict[node][u] * nbr_dict[node][v] * nbr_dict[u][v]) ** (1.0 / 3.0)
                    triangle_strength += w_uv_node / max_w
        denom = k * (k - 1) / 2.0
        return triangle_strength / denom if denom > 0 else 0.0

    logical_cc = {}
    for q in range(N):
        if q in logical_neighbors:
            logical_cc[q] = weighted_clustering(q, logical_neighbors)
        else:
            logical_cc[q] = 0.0

    # --- 3b. Unweighted local clustering coefficient (physical) ---
    physical_cc = {}
    for p in range(N):
        nbrs = physical_neighbors.get(p, set())
        k = len(nbrs)
        if k < 2:
            physical_cc[p] = 0.0
            continue
        triangles = 0
        nbr_list = list(nbrs)
        for i in range(len(nbr_list)):
            for j in range(i + 1, len(nbr_list)):
                if nbr_list[j] in physical_neighbors.get(nbr_list[i], set()):
                    triangles += 1
        denom = k * (k - 1) / 2.0
        physical_cc[p] = triangles / denom if denom > 0 else 0.0

    # --- 4. Sort and pair ---
    # Logical priority: active qubits with highest cc first; inactive last
    logical_order = sorted(
        range(N),
        key=lambda q: (q not in logical_qubits, -logical_cc[q], q)
    )
    # Physical priority: highest cc first; tie-break by degree desc, then id
    physical_order = sorted(
        range(N),
        key=lambda p: (-physical_cc[p], -len(physical_neighbors.get(p, set())), p)
    )

    mapping = [-1] * N
    reverse_mapping = [-1] * N
    used_physical = set()
    used_logical = set()

    for lq, pq in zip(logical_order, physical_order):
        if lq in used_logical or pq in used_physical:
            continue
        mapping[lq] = pq
        reverse_mapping[pq] = lq
        used_logical.add(lq)
        used_physical.add(pq)

    # --- 5. Identity fallback for any leftovers ---
    remaining_physical = [p for p in range(N) if p not in used_physical]
    rp_iter = iter(remaining_physical)
    for lq in range(N):
        if mapping[lq] == -1:
            try:
                pq = next(rp_iter)
            except StopIteration:
                break
            mapping[lq] = pq
            reverse_mapping[pq] = lq

    # Final safety: any still-unassigned slot gets identity-style fill
    for lq in range(N):
        if mapping[lq] == -1:
            for pq in range(N):
                if pq not in used_physical and reverse_mapping[pq] == -1:
                    mapping[lq] = pq
                    reverse_mapping[pq] = lq
                    used_physical.add(pq)
                    break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)