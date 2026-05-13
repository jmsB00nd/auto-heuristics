def init_mapping(self):
    import math
    import random
    from collections import defaultdict

    N = self.num_qubits
    D = self.distance_matrix

    # 1) Collect 2-qubit interactions from self.access (access2q may be None at init time)
    edge_w = defaultdict(int)
    active = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edge_w[(a, b)] += 1
            active.add(a)
            active.add(b)

    # Per-logical adjacency for incremental energy updates
    adj = defaultdict(list)  # q -> list of (neighbor, weight)
    edges = []
    for (a, b), w in edge_w.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
        edges.append((a, b, w))

    # 2) Warm-start permutation pi: pi[logical] = physical
    pi = list(range(N))
    try:
        from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
        md, _rmd = generate_structure_aware_initial_mapping(
            self.access, self.backend, self.distance_matrix, self.num_qubits
        )
        if md is not None and len(md) == N:
            seen = set()
            tmp = [None] * N
            for L in range(N):
                p = md[L] if L < len(md) else None
                if p is None or not (0 <= p < N) or p in seen:
                    continue
                tmp[L] = p
                seen.add(p)
            free = [p for p in range(N) if p not in seen]
            fi = 0
            for L in range(N):
                if tmp[L] is None:
                    tmp[L] = free[fi]
                    fi += 1
            pi = tmp
    except Exception:
        pi = list(range(N))

    # 3) Energy function
    def total_energy(perm):
        e = 0.0
        for (a, b, w) in edges:
            e += w * D[perm[a]][perm[b]]
        return e

    def local_energy(perm, q):
        e = 0.0
        pq = perm[q]
        for (nb, w) in adj[q]:
            e += w * D[pq][perm[nb]]
        return e

    cur_E = total_energy(pi)
    best_pi = list(pi)
    best_E = cur_E

    # 4) Simulated annealing
    if not edges or len(active) < 2:
        # Nothing meaningful to optimize; keep pi as is
        pass
    else:
        rng = random.Random(0xC0FFEE)
        active_list = list(active)
        # Iteration budget scales with problem size
        iters = max(2000, 50 * N + 20 * len(edges))
        iters = min(iters, 50000)
        T0 = max(1.0, 0.5 * (cur_E / max(1, len(edges))) + 1.0)
        T_end = 1e-3
        cooling = (T_end / T0) ** (1.0 / max(1, iters - 1))
        T = T0

        for _ in range(iters):
            # Pick i from active logicals; j from any logical (allows moving to idle physical)
            i = active_list[rng.randrange(len(active_list))]
            j = rng.randrange(N)
            if j == i:
                j = (j + 1) % N
            # Energy contribution involving i or j before swap
            before = 0.0
            for (nb, w) in adj[i]:
                if nb == j:
                    before += w * D[pi[i]][pi[j]]
                else:
                    before += w * D[pi[i]][pi[nb]]
            for (nb, w) in adj[j]:
                if nb == i:
                    continue  # already counted
                before += w * D[pi[j]][pi[nb]]
            # Swap
            pi[i], pi[j] = pi[j], pi[i]
            after = 0.0
            for (nb, w) in adj[i]:
                if nb == j:
                    after += w * D[pi[i]][pi[j]]
                else:
                    after += w * D[pi[i]][pi[nb]]
            for (nb, w) in adj[j]:
                if nb == i:
                    continue
                after += w * D[pi[j]][pi[nb]]
            dE = after - before
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
                cur_E += dE
                if cur_E < best_E:
                    best_E = cur_E
                    best_pi = list(pi)
            else:
                # Reject: undo
                pi[i], pi[j] = pi[j], pi[i]
            T *= cooling

        pi = best_pi

    # 5) Write outputs as length-N lists
    self.mapping_dict = list(pi)
    self.reverse_mapping_dict = [0] * N
    for L in range(N):
        self.reverse_mapping_dict[self.mapping_dict[L]] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)