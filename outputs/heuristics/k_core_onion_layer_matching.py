def init_mapping(self):
    import heapq

    N = self.num_qubits

    def compute_coreness(neighbors):
        deg = {v: len(neighbors[v]) for v in neighbors}
        coreness = {}
        processed = set()
        heap = [(deg[v], v) for v in neighbors]
        heapq.heapify(heap)
        while heap:
            d, v = heapq.heappop(heap)
            if v in processed or d != deg[v]:
                continue
            coreness[v] = d
            processed.add(v)
            for u in neighbors[v]:
                if u in processed:
                    continue
                if deg[u] > d:
                    deg[u] -= 1
                    heapq.heappush(heap, (deg[u], u))
        return coreness

    logical_neighbors = {}
    logical_qubits_seen = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_qubits_seen.add(a)
            logical_qubits_seen.add(b)
            logical_neighbors.setdefault(a, set()).add(b)
            logical_neighbors.setdefault(b, set()).add(a)
    for q in logical_qubits_seen:
        logical_neighbors.setdefault(q, set())

    activity = {}
    log_act = getattr(self, "logical_activity", None)
    qig = getattr(self, "qubit_interaction_graph", None)
    for q in logical_qubits_seen:
        if log_act is not None:
            try:
                activity[q] = log_act[q]
            except Exception:
                activity[q] = 0
        elif qig is not None:
            try:
                activity[q] = sum(qig[q].values())
            except Exception:
                activity[q] = len(logical_neighbors[q])
        else:
            activity[q] = len(logical_neighbors[q])

    backend = self.backend
    physical_neighbors = {}
    for p in range(N):
        try:
            nbrs = backend[p]
        except Exception:
            nbrs = ()
        physical_neighbors[p] = {u for u in nbrs if isinstance(u, int) and 0 <= u < N and u != p}

    logical_coreness = compute_coreness(logical_neighbors) if logical_neighbors else {}
    physical_coreness = compute_coreness(physical_neighbors)

    phys_cent = getattr(self, "physical_centrality", None) or {}

    logicals_sorted = sorted(
        [q for q in logical_qubits_seen if 0 <= q < N],
        key=lambda q: (-logical_coreness.get(q, 0), -activity.get(q, 0), q),
    )
    physicals_sorted = sorted(
        range(N),
        key=lambda p: (-physical_coreness.get(p, 0), -float(phys_cent.get(p, 0.0)), p),
    )

    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = set()
    used_log = set()

    pi = 0
    for log_q in logicals_sorted:
        if log_q in used_log:
            continue
        while pi < len(physicals_sorted) and physicals_sorted[pi] in used_phys:
            pi += 1
        if pi >= len(physicals_sorted):
            break
        phys_q = physicals_sorted[pi]
        mapping[log_q] = phys_q
        reverse[phys_q] = log_q
        used_phys.add(phys_q)
        used_log.add(log_q)
        pi += 1

    unused_phys = set(range(N)) - used_phys
    for q in range(N):
        if mapping[q] != -1:
            continue
        if q in unused_phys:
            mapping[q] = q
            reverse[q] = q
            unused_phys.discard(q)

    leftover = sorted(unused_phys)
    li = 0
    for q in range(N):
        if mapping[q] != -1:
            continue
        if li >= len(leftover):
            break
        p = leftover[li]
        mapping[q] = p
        reverse[p] = q
        li += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)