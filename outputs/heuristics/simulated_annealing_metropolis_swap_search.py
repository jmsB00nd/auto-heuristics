def init_mapping(self):
    import math
    import random

    N = self.num_qubits

    interactions = []
    logical_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            if q1 != q2:
                interactions.append((q1, q2))
            logical_set.add(q1)
            logical_set.add(q2)
        elif len(qubits) == 1:
            logical_set.add(qubits[0])

    for q in range(N):
        logical_set.add(q)

    qig = self.qubit_interaction_graph
    dmat = self.distance_matrix

    def edge_weight(a, b):
        try:
            return qig[a][b]
        except Exception:
            return 0

    weighted_pairs = []
    seen_pairs = set()
    for (a, b) in interactions:
        key = (a, b) if a < b else (b, a)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        w = edge_weight(a, b)
        if w <= 0:
            w = 1
        weighted_pairs.append((key[0], key[1], w))

    activity = self.logical_activity
    centrality = self.physical_centrality

    logicals_sorted = sorted(
        range(N),
        key=lambda q: (-activity.get(q, 0), q),
    )
    physicals_sorted = sorted(
        range(N),
        key=lambda p: (-centrality.get(p, 0.0), p),
    )

    mapping = [-1] * N
    inverse = [-1] * N
    used_phys = set()
    pi = 0
    for lq in logicals_sorted:
        while pi < N and physicals_sorted[pi] in used_phys:
            pi += 1
        if pi >= N:
            break
        pq = physicals_sorted[pi]
        mapping[lq] = pq
        inverse[pq] = lq
        used_phys.add(pq)
        pi += 1

    remaining_phys = [p for p in range(N) if p not in used_phys]
    rp_idx = 0
    for lq in range(N):
        if mapping[lq] == -1:
            if rp_idx < len(remaining_phys):
                pq = remaining_phys[rp_idx]
                rp_idx += 1
                mapping[lq] = pq
                inverse[pq] = lq

    for lq in range(N):
        if mapping[lq] == -1:
            for pq in range(N):
                if inverse[pq] == -1:
                    mapping[lq] = pq
                    inverse[pq] = lq
                    break

    incident = [[] for _ in range(N)]
    for (a, b, w) in weighted_pairs:
        incident[a].append((b, w))
        incident[b].append((a, w))

    def total_cost(mp):
        c = 0.0
        for (a, b, w) in weighted_pairs:
            c += w * dmat[mp[a]][mp[b]]
        return c

    def swap_delta(mp, la, lb):
        if la == lb:
            return 0.0
        pa = mp[la]
        pb = mp[lb]
        delta = 0.0
        for (nb, w) in incident[la]:
            if nb == lb:
                continue
            pn = mp[nb]
            delta += w * (dmat[pb][pn] - dmat[pa][pn])
        for (nb, w) in incident[lb]:
            if nb == la:
                continue
            pn = mp[nb]
            delta += w * (dmat[pa][pn] - dmat[pb][pn])
        return delta

    rng = random.Random(0xC0FFEE)

    cur_cost = total_cost(mapping)
    best_mapping = list(mapping)
    best_cost = cur_cost

    if N >= 2 and len(weighted_pairs) > 0:
        M = max(1, len(weighted_pairs))
        iterations = min(20000, max(500, 50 * N + 10 * M))
        T0 = max(1.0, abs(cur_cost) / max(1.0, M) * 2.0)
        T_end = 1e-3
        alpha = (T_end / T0) ** (1.0 / max(1, iterations - 1)) if T0 > T_end else 0.99

        T = T0
        for _ in range(iterations):
            la = rng.randrange(N)
            lb = rng.randrange(N)
            if la == lb:
                T *= alpha
                continue
            d = swap_delta(mapping, la, lb)
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
                pa = mapping[la]
                pb = mapping[lb]
                mapping[la] = pb
                mapping[lb] = pa
                inverse[pa] = lb
                inverse[pb] = la
                cur_cost += d
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_mapping = list(mapping)
            T *= alpha

    final_mapping = best_mapping
    final_inverse = [-1] * N
    for lq in range(N):
        final_inverse[final_mapping[lq]] = lq

    self.mapping_dict = list(final_mapping)
    self.reverse_mapping_dict = list(final_inverse)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)