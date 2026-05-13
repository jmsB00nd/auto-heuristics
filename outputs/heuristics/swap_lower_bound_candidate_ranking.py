def init_mapping(self):
    import random

    N = self.num_qubits

    interactions = []
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a != b:
                interactions.append((a, b))

    dist = self.distance_matrix

    def qig_w(a, b):
        inner = self.qubit_interaction_graph.get(a) if hasattr(self.qubit_interaction_graph, "get") else None
        if inner is None:
            try:
                return self.qubit_interaction_graph[a][b]
            except Exception:
                return 0
        return inner.get(b, 0)

    def activity(q):
        if hasattr(self.logical_activity, "get"):
            return self.logical_activity.get(q, 0)
        try:
            return self.logical_activity[q]
        except Exception:
            return 0

    def centrality(p):
        if hasattr(self.physical_centrality, "get"):
            return self.physical_centrality.get(p, 0.0)
        try:
            return self.physical_centrality[p]
        except Exception:
            return 0.0

    def swap_lower_bound(mapping):
        total = 0
        for a, b in interactions:
            d = dist[mapping[a]][mapping[b]]
            if d > 1:
                total += d - 1
        return total

    def is_valid_perm(m):
        if m is None or len(m) != N:
            return False
        if any((p is None or p < 0 or p >= N) for p in m):
            return False
        return len(set(m)) == N

    def cand_identity():
        return list(range(N))

    def cand_activity_centrality(reverse=False):
        logicals = sorted(range(N), key=lambda q: (-activity(q), q))
        physicals = sorted(range(N), key=lambda p: (-centrality(p), p))
        if reverse:
            physicals = list(reversed(physicals))
        m = [0] * N
        for lg, ph in zip(logicals, physicals):
            m[lg] = ph
        return m

    def cand_greedy_qig():
        m = [-1] * N
        used_phys = [False] * N
        placed = [False] * N

        logicals_by_act = sorted(range(N), key=lambda q: (-activity(q), q))
        physicals_by_cent = sorted(range(N), key=lambda p: (-centrality(p), p))

        seed_log = logicals_by_act[0]
        seed_phys = physicals_by_cent[0]
        m[seed_log] = seed_phys
        used_phys[seed_phys] = True
        placed[seed_log] = True

        for _ in range(N - 1):
            best_log, best_score = -1, -1
            for q in range(N):
                if placed[q]:
                    continue
                s = 0
                for qp in range(N):
                    if placed[qp]:
                        s += qig_w(q, qp)
                if s > best_score or (s == best_score and best_log == -1):
                    best_score = s
                    best_log = q
            if best_log < 0:
                break
            if best_score <= 0:
                for q in logicals_by_act:
                    if not placed[q]:
                        best_log = q
                        break

            best_phys, best_cost = -1, float("inf")
            for p in range(N):
                if used_phys[p]:
                    continue
                cost = 0.0
                for qp in range(N):
                    if placed[qp]:
                        w = qig_w(best_log, qp)
                        if w:
                            cost += w * dist[p][m[qp]]
                cost -= 1e-9 * centrality(p)
                if cost < best_cost:
                    best_cost = cost
                    best_phys = p
            if best_phys < 0:
                for p in physicals_by_cent:
                    if not used_phys[p]:
                        best_phys = p
                        break
            m[best_log] = best_phys
            used_phys[best_phys] = True
            placed[best_log] = True

        for q in range(N):
            if not placed[q]:
                for p in physicals_by_cent:
                    if not used_phys[p]:
                        m[q] = p
                        used_phys[p] = True
                        placed[q] = True
                        break
        return m

    def cand_random(seed):
        rng = random.Random(seed)
        perm = list(range(N))
        rng.shuffle(perm)
        return perm

    candidates = []
    for builder in (cand_identity,
                    lambda: cand_activity_centrality(False),
                    lambda: cand_activity_centrality(True),
                    cand_greedy_qig):
        try:
            c = builder()
            if is_valid_perm(c):
                candidates.append(c)
        except Exception:
            pass

    for seed in (1, 7, 19, 41, 97, 211, 523, 1031):
        try:
            c = cand_random(seed)
            if is_valid_perm(c):
                candidates.append(c)
        except Exception:
            pass

    if not candidates:
        candidates.append(list(range(N)))

    best = min(candidates, key=swap_lower_bound)

    self.mapping_dict = list(best)
    self.reverse_mapping_dict = [0] * N
    for lg in range(N):
        self.reverse_mapping_dict[self.mapping_dict[lg]] = lg

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)