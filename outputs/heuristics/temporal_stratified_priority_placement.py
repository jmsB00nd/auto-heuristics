def init_mapping(self):
    import math
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    # Step 1: collect 2q interactions in temporal order
    interactions = []
    try:
        sorted_gate_ids = sorted(self.access.keys())
    except Exception:
        sorted_gate_ids = list(self.access.keys())

    depth_idx = 0
    for gid in sorted_gate_ids:
        qubits = self.access[gid]
        if qubits is None:
            continue
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            if q1 != q2 and 0 <= q1 < N and 0 <= q2 < N:
                interactions.append((q1, q2, depth_idx))
                depth_idx += 1

    M = len(interactions)

    # Step 2: weighted interaction graph with temporal decay
    alpha = 3.0 / max(M, 1)
    weighted_graph = defaultdict(lambda: defaultdict(float))
    logical_qubits = set()
    for q1, q2, d in interactions:
        w = math.exp(-alpha * d)
        weighted_graph[q1][q2] += w
        weighted_graph[q2][q1] += w
        logical_qubits.add(q1)
        logical_qubits.add(q2)

    weighted_deg = {q: sum(weighted_graph[q].values()) for q in logical_qubits}

    # Step 3: physical centrality from distance_matrix
    phys_centrality = [0.0] * N
    for p in range(N):
        s = 0.0
        for q in range(N):
            if p == q:
                continue
            try:
                d = self.distance_matrix[p][q]
            except Exception:
                d = 0
            if d and d > 0:
                s += 1.0 / d
        phys_centrality[p] = s

    used_phys = set()
    placed_logical = set()

    if logical_qubits:
        seed_logical = max(logical_qubits, key=lambda q: (weighted_deg[q], -q))
        seed_phys = max(range(N), key=lambda p: (phys_centrality[p], -p))
        self.mapping_dict[seed_logical] = seed_phys
        self.reverse_mapping_dict[seed_phys] = seed_logical
        used_phys.add(seed_phys)
        placed_logical.add(seed_logical)

        remaining = [q for q in logical_qubits if q not in placed_logical]

        # Step 4: iterative weighted greedy placement
        while remaining:
            def priority(q):
                neigh = weighted_graph[q]
                return sum(neigh[lp] for lp in placed_logical if lp in neigh)
            remaining.sort(key=lambda q: (-priority(q), -weighted_deg.get(q, 0.0), q))
            next_logical = remaining.pop(0)

            best_phys = None
            best_cost = float('inf')
            neigh = weighted_graph[next_logical]
            for p in range(N):
                if p in used_phys:
                    continue
                cost = 0.0
                for lp in placed_logical:
                    w = neigh.get(lp, 0.0)
                    if w > 0.0:
                        try:
                            d = self.distance_matrix[p][self.mapping_dict[lp]]
                        except Exception:
                            d = 0
                        cost += w * d
                if cost < best_cost:
                    best_cost = cost
                    best_phys = p

            if best_phys is None:
                break

            self.mapping_dict[next_logical] = best_phys
            self.reverse_mapping_dict[best_phys] = next_logical
            used_phys.add(best_phys)
            placed_logical.add(next_logical)

    # Step 5: identity fallback for the rest
    unused_phys = [p for p in range(N) if p not in used_phys]
    unused_set = set(unused_phys)
    for L in range(N):
        if self.mapping_dict[L] is None:
            if L in unused_set:
                p = L
                unused_phys.remove(L)
                unused_set.discard(L)
            else:
                p = unused_phys.pop(0)
                unused_set.discard(p)
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L
            used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)