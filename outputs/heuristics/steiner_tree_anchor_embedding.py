def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    edge_weight = defaultdict(int)
    logical_activity = defaultdict(int)
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_qubits.add(a); logical_qubits.add(b)
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_weight[key] += 1
            logical_activity[a] += 1
            logical_activity[b] += 1
        else:
            for q in qubits:
                logical_qubits.add(q)

    total_weight = sum(edge_weight.values())
    X_PCT = 0.7
    target = X_PCT * total_weight

    sorted_logicals = sorted(logical_qubits, key=lambda q: -logical_activity.get(q, 0))
    core_set = set()
    if total_weight > 0:
        induced = 0
        for q in sorted_logicals:
            for r in core_set:
                key = (q, r) if q < r else (r, q)
                if key in edge_weight:
                    induced += edge_weight[key]
            core_set.add(q)
            if induced >= target:
                break
    core_size = max(len(core_set), 1)

    centrality = getattr(self, "physical_centrality", None) or {}
    def cscore(p):
        return centrality.get(p, 0.0)

    if N > 0:
        seed = max(range(N), key=cscore)
    else:
        seed = 0

    selected = {seed}
    frontier = set(self.backend.get(seed, set())) - selected
    while len(selected) < core_size and frontier:
        best = max(frontier, key=cscore)
        selected.add(best)
        frontier.discard(best)
        for nb in self.backend.get(best, set()):
            if nb not in selected:
                frontier.add(nb)
    if len(selected) < core_size:
        for p in sorted(range(N), key=cscore, reverse=True):
            if len(selected) >= core_size:
                break
            selected.add(p)

    mapping = [None] * N
    reverse_mapping = [None] * N
    used_phys = set()
    used_log = set()

    sorted_core = sorted(core_set, key=lambda q: -logical_activity.get(q, 0))
    sorted_phys = sorted(selected, key=lambda p: -cscore(p))
    for i, l in enumerate(sorted_core):
        if i >= len(sorted_phys) or l >= N:
            break
        p = sorted_phys[i]
        if p in used_phys or mapping[l] is not None:
            continue
        mapping[l] = p
        reverse_mapping[p] = l
        used_phys.add(p); used_log.add(l)

    non_core = [q for q in logical_qubits if q not in used_log and q < N]
    non_core.sort(key=lambda q: -logical_activity.get(q, 0))
    residual = [p for p in range(N) if p not in used_phys]
    residual.sort(key=cscore, reverse=True)
    ri = 0
    for l in non_core:
        while ri < len(residual) and residual[ri] in used_phys:
            ri += 1
        if ri >= len(residual):
            break
        p = residual[ri]; ri += 1
        if mapping[l] is not None:
            continue
        mapping[l] = p
        reverse_mapping[p] = l
        used_phys.add(p); used_log.add(l)

    leftover = [p for p in range(N) if p not in used_phys]
    li = 0
    for l in range(N):
        if mapping[l] is None:
            if l not in used_phys:
                p = l
            else:
                while li < len(leftover) and leftover[li] in used_phys:
                    li += 1
                if li >= len(leftover):
                    continue
                p = leftover[li]; li += 1
            mapping[l] = p
            reverse_mapping[p] = l
            used_phys.add(p)

    if any(m is None for m in mapping):
        free = [p for p in range(N) if p not in used_phys]
        fi = 0
        for l in range(N):
            if mapping[l] is None and fi < len(free):
                p = free[fi]; fi += 1
                mapping[l] = p
                reverse_mapping[p] = l
                used_phys.add(p)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)