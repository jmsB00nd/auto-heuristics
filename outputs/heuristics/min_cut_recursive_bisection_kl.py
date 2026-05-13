def init_mapping(self):
    from collections import defaultdict
    import random

    N = self.num_qubits

    qig_w = defaultdict(lambda: defaultdict(float))
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            if q1 == q2:
                continue
            qig_w[q1][q2] += 1.0
            qig_w[q2][q1] += 1.0

    hw_w = defaultdict(lambda: defaultdict(float))
    for p in range(N):
        for q in self.backend.get(p, ()):
            if p != q:
                hw_w[p][q] = 1.0

    rng = random.Random(0xC0FFEE)

    def kl_bisect(nodes, weights):
        nodes = list(nodes)
        n = len(nodes)
        if n <= 1:
            return nodes, []
        half = n // 2
        shuffled = list(nodes)
        rng.shuffle(shuffled)
        A = set(shuffled[:half])
        B = set(shuffled[half:])

        def w(u, v):
            row = weights.get(u)
            if row is None:
                return 0.0
            return row.get(v, 0.0)

        for _outer in range(6):
            D = {}
            for u in A:
                ext = sum(w(u, v) for v in B)
                inn = sum(w(u, v) for v in A if v != u)
                D[u] = ext - inn
            for u in B:
                ext = sum(w(u, v) for v in A)
                inn = sum(w(u, v) for v in B if v != u)
                D[u] = ext - inn

            A_free = set(A)
            B_free = set(B)
            gains = []
            swaps = []
            steps = min(len(A_free), len(B_free))
            for _s in range(steps):
                best_gain = None
                best_pair = None
                for a in A_free:
                    da = D[a]
                    for b in B_free:
                        g = da + D[b] - 2.0 * w(a, b)
                        if best_gain is None or g > best_gain:
                            best_gain = g
                            best_pair = (a, b)
                if best_pair is None:
                    break
                a, b = best_pair
                gains.append(best_gain)
                swaps.append((a, b))
                A_free.discard(a)
                B_free.discard(b)
                for x in A_free:
                    D[x] = D[x] + 2.0 * w(x, a) - 2.0 * w(x, b)
                for x in B_free:
                    D[x] = D[x] + 2.0 * w(x, b) - 2.0 * w(x, a)

            best_k = 0
            best_sum = 0.0
            cum = 0.0
            for k, g in enumerate(gains, 1):
                cum += g
                if cum > best_sum + 1e-12:
                    best_sum = cum
                    best_k = k

            if best_k == 0:
                break

            for i in range(best_k):
                a, b = swaps[i]
                A.discard(a)
                B.discard(b)
                A.add(b)
                B.add(a)

        return list(A), list(B)

    def recursive_bisect(nodes, weights):
        nodes = list(nodes)
        if len(nodes) <= 1:
            return nodes
        A, B = kl_bisect(nodes, weights)
        if not A or not B:
            mid = len(nodes) // 2
            A, B = nodes[:mid], nodes[mid:]
        return recursive_bisect(A, weights) + recursive_bisect(B, weights)

    try:
        logical_order = recursive_bisect(list(range(N)), qig_w)
        physical_order = recursive_bisect(list(range(N)), hw_w)
        if len(logical_order) != N or len(physical_order) != N \
                or len(set(logical_order)) != N or len(set(physical_order)) != N:
            raise RuntimeError("bisection orderings malformed")
    except Exception:
        logical_order = list(range(N))
        physical_order = list(range(N))

    mapping = [0] * N
    reverse = [0] * N
    used_phys = set()
    used_log = set()
    for log_q, phys_q in zip(logical_order, physical_order):
        mapping[log_q] = phys_q
        reverse[phys_q] = log_q
        used_phys.add(phys_q)
        used_log.add(log_q)

    if len(used_phys) != N or len(used_log) != N:
        free_phys = [p for p in range(N) if p not in used_phys]
        idx = 0
        for L in range(N):
            if L not in used_log:
                P = free_phys[idx]
                idx += 1
                mapping[L] = P
                reverse[P] = L

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)