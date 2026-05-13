def init_mapping(self):
    N = self.num_qubits

    logical_set = set()
    pair_weights = {}
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_set.add(a)
            logical_set.add(b)
            key = (a, b) if a <= b else (b, a)
            pair_weights[key] = pair_weights.get(key, 0) + 1
        elif len(qubits) == 1:
            logical_set.add(qubits[0])

    weighted_degree = {}
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig is not None:
        for q in logical_set:
            row = qig.get(q, {}) if hasattr(qig, "get") else qig[q]
            total = 0
            for nbr, w in row.items():
                total += w
            weighted_degree[q] = total
    if not weighted_degree or all(v == 0 for v in weighted_degree.values()):
        weighted_degree = {q: 0 for q in logical_set}
        for (a, b), w in pair_weights.items():
            weighted_degree[a] = weighted_degree.get(a, 0) + w
            weighted_degree[b] = weighted_degree.get(b, 0) + w

    logical_activity = getattr(self, "logical_activity", {})

    logical_ranked = sorted(
        logical_set,
        key=lambda q: (-weighted_degree.get(q, 0),
                       -logical_activity.get(q, 0) if hasattr(logical_activity, "get") else 0,
                       q),
    )

    centrality = getattr(self, "physical_centrality", {}) or {}
    physical_ranked = sorted(
        range(N),
        key=lambda p: (-centrality.get(p, 0.0), p),
    )

    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = set()

    pair_count = min(len(logical_ranked), len(physical_ranked))
    for i in range(pair_count):
        L = logical_ranked[i]
        if L >= N:
            continue
        P = physical_ranked[i]
        mapping[L] = P
        reverse[P] = L
        used_phys.add(P)

    remaining_logicals = [L for L in range(N) if mapping[L] == -1]
    remaining_physicals = [P for P in physical_ranked if P not in used_phys]
    leftover_central = [P for P in remaining_physicals]
    idx = 0
    for L in remaining_logicals:
        if idx >= len(leftover_central):
            break
        P = leftover_central[idx]
        idx += 1
        mapping[L] = P
        reverse[P] = L
        used_phys.add(P)

    for L in range(N):
        if mapping[L] == -1:
            for P in range(N):
                if P not in used_phys:
                    mapping[L] = P
                    reverse[P] = L
                    used_phys.add(P)
                    break

    dist = self.distance_matrix

    def edge_cost(u, v):
        pu = mapping[u]
        pv = mapping[v]
        if pu < 0 or pv < 0:
            return 0
        return dist[pu][pv]

    edges = []
    for (a, b), w in pair_weights.items():
        if a < N and b < N and mapping[a] >= 0 and mapping[b] >= 0:
            edges.append((a, b, w))

    incident = {}
    for a, b, w in edges:
        incident.setdefault(a, []).append((b, w))
        incident.setdefault(b, []).append((a, w))

    def local_cost(q):
        c = 0
        for nbr, w in incident.get(q, []):
            c += w * edge_cost(q, nbr)
        return c

    seen_pairs = set()
    for a, b, w in edges:
        key = (a, b) if a < b else (b, a)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pa, pb = mapping[a], mapping[b]
        if pa < 0 or pb < 0:
            continue
        before = local_cost(a) + local_cost(b)
        shared = w * dist[pa][pb]
        before -= shared
        mapping[a], mapping[b] = pb, pa
        reverse[pa], reverse[pb] = b, a
        after = local_cost(a) + local_cost(b)
        shared_after = w * dist[mapping[a]][mapping[b]]
        after -= shared_after
        if after + shared_after >= before + shared:
            mapping[a], mapping[b] = pa, pb
            reverse[pa], reverse[pb] = a, b

    for L in range(N):
        if mapping[L] == -1:
            for P in range(N):
                if reverse[P] == -1:
                    mapping[L] = P
                    reverse[P] = L
                    break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)