def init_mapping(self):
    import random
    from collections import defaultdict

    N = self.num_qubits

    # ---- warm start ----
    mapping = list(range(N))
    reverse = list(range(N))
    try:
        from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
        m0, r0 = generate_structure_aware_initial_mapping(
            self.access, self.backend, self.distance_matrix, N
        )
        if isinstance(m0, list) and isinstance(r0, list) and len(m0) == N and len(r0) == N:
            mapping = list(m0)
            reverse = list(r0)
    except Exception:
        pass

    # repair if warm-start is malformed
    if len(set(mapping)) != N or any(reverse[mapping[l]] != l for l in range(N)):
        mapping = list(range(N))
        reverse = list(range(N))

    D = self.distance_matrix
    QIG = self.qubit_interaction_graph

    # collect logicals that actually interact
    active = set()
    for gid, qs in self.access.items():
        if len(qs) == 2:
            active.add(qs[0]); active.add(qs[1])
    active_logicals = sorted(active) if active else list(range(N))

    def total_cost(m):
        c = 0.0
        seen = set()
        for u in active_logicals:
            for v, w in QIG[u].items():
                if v == u: continue
                key = (u, v) if u < v else (v, u)
                if key in seen: continue
                seen.add(key)
                c += w * D[m[u]][m[v]]
        return c

    def cycle_delta(m, cycle):
        k = len(cycle)
        old_phys = [m[l] for l in cycle]
        new_phys = [old_phys[(i + 1) % k] for i in range(k)]
        cycle_set = set(cycle)
        delta = 0.0
        internal_done = set()
        for i, li in enumerate(cycle):
            for nb, w in QIG[li].items():
                if nb == li: continue
                if nb in cycle_set:
                    pair = (li, nb) if li < nb else (nb, li)
                    if pair in internal_done: continue
                    internal_done.add(pair)
                    j = cycle.index(nb)
                    delta += w * (D[new_phys[i]][new_phys[j]] - D[old_phys[i]][old_phys[j]])
                else:
                    pnb = m[nb]
                    delta += w * (D[new_phys[i]][pnb] - D[old_phys[i]][pnb])
        return delta

    def apply_cycle(m, rev, cycle):
        k = len(cycle)
        old_phys = [m[l] for l in cycle]
        for i, l in enumerate(cycle):
            np_i = old_phys[(i + 1) % k]
            m[l] = np_i
            rev[np_i] = l

    current_cost = total_cost(mapping)
    best_mapping = list(mapping)
    best_cost = current_cost

    tabu = {}  # (logical, physical) -> expiry iter
    tabu_tenure = max(5, N // 4)
    max_iters = max(60, 5 * N)
    no_improve_limit = max(25, 2 * N)
    rng = random.Random(0xC0FFEE)

    iteration = 0
    no_improve = 0

    while iteration < max_iters and no_improve < no_improve_limit:
        iteration += 1
        best_move = None
        best_delta = float('inf')

        # ---- 2-opt chain (swap) neighborhood ----
        sample = active_logicals if len(active_logicals) <= 16 else rng.sample(active_logicals, 16)
        for li in sample:
            cand = set()
            for nb in QIG[li].keys():
                if nb == li: continue
                pnb = mapping[nb]
                cand.add(pnb)
                cand.update(self.backend[pnb])
            for _ in range(2):
                cand.add(rng.randrange(N))
            for new_phys in cand:
                if new_phys == mapping[li]: continue
                lj = reverse[new_phys]
                if lj == li: continue
                old_pi = mapping[li]
                blocked = ((li, old_pi) in tabu and tabu[(li, old_pi)] > iteration) or \
                          ((lj, new_phys) in tabu and tabu[(lj, new_phys)] > iteration)
                d = cycle_delta(mapping, [li, lj])
                if blocked and (current_cost + d) >= best_cost:
                    continue
                if d < best_delta:
                    best_delta = d
                    best_move = [li, lj]

        # ---- 3-opt chain relocations (k=3) for diversification ----
        if len(active_logicals) >= 3:
            for _ in range(min(12, len(active_logicals))):
                trio = rng.sample(active_logicals, 3)
                d = cycle_delta(mapping, trio)
                k = 3
                old_phys = [mapping[l] for l in trio]
                blocked = False
                for i, l in enumerate(trio):
                    if (l, old_phys[i]) in tabu and tabu[(l, old_phys[i])] > iteration:
                        blocked = True; break
                if blocked and (current_cost + d) >= best_cost:
                    continue
                if d < best_delta:
                    best_delta = d
                    best_move = list(trio)

        if best_move is None:
            break

        # apply chosen chain relocation
        k = len(best_move)
        old_phys = [mapping[l] for l in best_move]
        for i, l in enumerate(best_move):
            tabu[(l, old_phys[i])] = iteration + tabu_tenure
        apply_cycle(mapping, reverse, best_move)
        current_cost += best_delta

        if current_cost < best_cost - 1e-12:
            best_cost = current_cost
            best_mapping = list(mapping)
            no_improve = 0
        else:
            no_improve += 1

        if iteration % 25 == 0:
            tabu = {kk: vv for kk, vv in tabu.items() if vv > iteration}

    # restore best and rebuild reverse from scratch
    mapping = list(best_mapping)
    if len(mapping) != N or len(set(mapping)) != N:
        mapping = list(range(N))
    reverse = [0] * N
    for l in range(N):
        reverse[mapping[l]] = l

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)