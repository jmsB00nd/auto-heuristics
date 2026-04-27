def init_mapping(self):
    import math
    from collections import defaultdict, deque

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    interactions = []
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a != b:
                interactions.append((a, b))

    logical_graph = defaultdict(lambda: defaultdict(float))
    logical_qubits_set = set()
    for (a, b) in interactions:
        logical_graph[a][b] += 1.0
        logical_graph[b][a] += 1.0
        logical_qubits_set.add(a)
        logical_qubits_set.add(b)

    logical_qubits = sorted(logical_qubits_set)

    def weighted_degree(q):
        return sum(logical_graph[q].values())

    def logical_bfs_dist(src, targets):
        if not targets:
            return {}
        dist = {src: 0}
        dq = deque([src])
        remaining = set(targets)
        remaining.discard(src)
        result = {src: 0}
        while dq and remaining:
            u = dq.popleft()
            for v in logical_graph[u].keys():
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
                    if v in remaining:
                        result[v] = dist[v]
                        remaining.discard(v)
        for t in targets:
            if t not in result:
                result[t] = len(logical_qubits) + 1
        return result

    K = max(2, min(8, int(math.ceil(math.log2(max(N, 2)))) + 1))
    K = min(K, N)

    try:
        ecc = [max(self.distance_matrix[p]) for p in range(N)]
        first_landmark = max(range(N), key=lambda p: ecc[p])
    except Exception:
        first_landmark = 0

    landmarks = [first_landmark]
    min_dist_to_landmarks = [self.distance_matrix[first_landmark][p] for p in range(N)]
    while len(landmarks) < K:
        next_lm = max(range(N), key=lambda p: min_dist_to_landmarks[p] if p not in landmarks else -1)
        if next_lm in landmarks:
            break
        landmarks.append(next_lm)
        for p in range(N):
            d = self.distance_matrix[next_lm][p]
            if d < min_dist_to_landmarks[p]:
                min_dist_to_landmarks[p] = d

    K_eff = len(landmarks)

    phys_sig = [None] * N
    for p in range(N):
        phys_sig[p] = tuple(self.distance_matrix[lm][p] for lm in landmarks)

    if logical_qubits:
        sorted_by_deg = sorted(logical_qubits, key=lambda q: -weighted_degree(q))
        anchors = sorted_by_deg[:K_eff]
        while len(anchors) < K_eff and len(anchors) < len(logical_qubits):
            for q in sorted_by_deg:
                if q not in anchors:
                    anchors.append(q)
                    if len(anchors) >= K_eff:
                        break
            break
    else:
        anchors = []

    log_sig = {}
    for q in logical_qubits:
        dists = logical_bfs_dist(q, anchors) if anchors else {}
        sig = []
        for a in anchors:
            w = logical_graph[q].get(a, 0.0)
            d = dists.get(a, len(logical_qubits) + 1)
            if q == a:
                sig.append(0.0)
            else:
                sig.append(float(d) / (1.0 + w))
        log_sig[q] = tuple(sig)

    if log_sig:
        all_log_vals = [v for sig in log_sig.values() for v in sig]
        all_phys_vals = [v for sig in phys_sig for v in sig]
        max_log = max(all_log_vals) if all_log_vals else 1.0
        max_phys = max(all_phys_vals) if all_phys_vals else 1.0
        scale = (max_phys / max_log) if max_log > 0 else 1.0
        for q in log_sig:
            log_sig[q] = tuple(v * scale for v in log_sig[q])

    order = sorted(logical_qubits, key=lambda q: -weighted_degree(q))

    used = set()

    def sig_dist(sig_a, sig_b):
        s = 0.0
        for x, y in zip(sig_a, sig_b):
            d = x - y
            s += d * d
        return s

    for q in order:
        if q >= N:
            continue
        lsig = log_sig.get(q, tuple([0.0] * K_eff))
        best_p = -1
        best_d = float("inf")
        for p in range(N):
            if p in used:
                continue
            d = sig_dist(lsig, phys_sig[p])
            if d < best_d:
                best_d = d
                best_p = p
        if best_p == -1:
            continue
        self.mapping_dict[q] = best_p
        self.reverse_mapping_dict[best_p] = q
        used.add(best_p)

    unused_physical = [p for p in range(N) if p not in used]
    up_iter = iter(unused_physical)
    for q in range(N):
        if self.mapping_dict[q] == -1:
            try:
                p = next(up_iter)
            except StopIteration:
                break
            self.mapping_dict[q] = p
            self.reverse_mapping_dict[p] = q
            used.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)