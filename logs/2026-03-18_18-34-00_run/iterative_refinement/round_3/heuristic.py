def init_mapping(self):
    """
    Layer-Decay Weighted Greedy + Simulated Annealing (LDWGSA)

    Improvements over WIGPLS:
    1. Layer-decayed weights: earlier gates get higher weight (critical path focus).
    2. Multi-seed: tries top-3 highest-interaction logical pairs as seeds.
    3. Simulated annealing: escapes local optima unreachable by deterministic 2-opt.
    4. Best-solution tracking during SA + final 2-opt cleanup.
    """
    import math
    import random
    from collections import defaultdict

    # ── Layer-decayed interaction weights ─────────────────────────────────
    # Earlier gates get weight decay^idx, so they dominate placement decisions.
    # This biases the mapping toward minimising swaps for the circuit's first
    # (often most parallelism-constraining) layers.
    logical_qubit_set = set()
    interaction_weight = defaultdict(lambda: defaultdict(float))
    total_iw = defaultdict(float)

    decay = 0.98
    gates = list(self.access.items())

    for idx, (gate, qubits) in enumerate(gates):
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            w = decay ** idx
            interaction_weight[q1][q2] += w
            interaction_weight[q2][q1] += w
            total_iw[q1] += w
            total_iw[q2] += w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # ── Fallback ──────────────────────────────────────────────────────────
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    hw_degree = {p: len(self.backend[p]) for p in physical_qubits}

    # ── Cost helpers ──────────────────────────────────────────────────────
    def total_cost(lq_to_phys):
        cost = 0.0
        for lq1, p1 in lq_to_phys.items():
            for lq2, w in interaction_weight[lq1].items():
                if lq2 > lq1 and lq2 in lq_to_phys:
                    p2 = lq_to_phys[lq2]
                    d = self.distance_matrix[p1][p2]
                    if d != float('inf'):
                        cost += w * d
        return cost

    def swap_delta(lq_to_phys, lq1, lq2):
        p1 = lq_to_phys[lq1]
        p2 = lq_to_phys[lq2]
        delta = 0.0
        for lq_o, w in interaction_weight[lq1].items():
            if lq_o == lq2 or lq_o not in lq_to_phys:
                continue
            p_o = lq_to_phys[lq_o]
            d_old = self.distance_matrix[p1][p_o]
            d_new = self.distance_matrix[p2][p_o]
            if d_old != float('inf') and d_new != float('inf'):
                delta += w * (d_new - d_old)
        for lq_o, w in interaction_weight[lq2].items():
            if lq_o == lq1 or lq_o not in lq_to_phys:
                continue
            p_o = lq_to_phys[lq_o]
            d_old = self.distance_matrix[p2][p_o]
            d_new = self.distance_matrix[p1][p_o]
            if d_old != float('inf') and d_new != float('inf'):
                delta += w * (d_new - d_old)
        return delta

    # ── Greedy placement from a given seed ────────────────────────────────
    def run_greedy(seed_assignments):
        lq_to_phys = dict(seed_assignments)
        placed_phys = set(lq_to_phys.values())
        unplaced = set(lq for lq in logical_qubits if lq not in lq_to_phys)

        while unplaced:
            best_lq = max(
                unplaced,
                key=lambda lq: (
                    sum(interaction_weight[lq].get(pl, 0) for pl in lq_to_phys),
                    total_iw[lq],
                ),
            )

            free_phys = [p for p in physical_qubits if p not in placed_phys]
            if not free_phys:
                break

            iw_cur = dict(interaction_weight[best_lq])
            snapshot = list(lq_to_phys.items())

            def placement_cost(p, _iw=iw_cur, _snap=snapshot):
                c = 0.0
                for pl_lq, pl_phys in _snap:
                    w = _iw.get(pl_lq, 0)
                    if w == 0:
                        continue
                    d = self.distance_matrix[p][pl_phys]
                    if d != float('inf'):
                        c += w * d
                return c

            best_phys = min(free_phys, key=placement_cost)
            lq_to_phys[best_lq] = best_phys
            placed_phys.add(best_phys)
            unplaced.discard(best_lq)

        return lq_to_phys

    # ── Multi-seed exploration ────────────────────────────────────────────
    seed_lq_candidates = sorted(
        (
            (w, lq1, lq2)
            for lq1 in logical_qubits
            for lq2, w in interaction_weight[lq1].items()
            if lq1 < lq2
        ),
        reverse=True,
    )

    hw_pair_candidates = sorted(
        (
            (self.distance_matrix[p1][p2], -(hw_degree[p1] + hw_degree[p2]), p1, p2)
            for p1 in physical_qubits
            for p2 in physical_qubits
            if p1 < p2 and self.distance_matrix[p1][p2] != float('inf')
        )
    )

    best_lq_to_phys = None
    best_greedy_cost = float('inf')

    n_seed_tries = min(3, len(seed_lq_candidates))

    if n_seed_tries > 0 and hw_pair_candidates:
        _, _, hp1, hp2 = hw_pair_candidates[0]
        for i in range(n_seed_tries):
            _, lq1_s, lq2_s = seed_lq_candidates[i]
            for p_a, p_b in [(hp1, hp2), (hp2, hp1)]:
                candidate = run_greedy({lq1_s: p_a, lq2_s: p_b})
                c = total_cost(candidate)
                if c < best_greedy_cost:
                    best_greedy_cost = c
                    best_lq_to_phys = candidate

    if best_lq_to_phys is None:
        if hw_pair_candidates:
            _, _, hp1, hp2 = hw_pair_candidates[0]
            seed = {logical_qubits[0]: hp1}
            if len(logical_qubits) > 1:
                seed[logical_qubits[1]] = hp2
        else:
            seed_phys = max(physical_qubits, key=lambda p: hw_degree[p])
            seed = {logical_qubits[0]: seed_phys}
        best_lq_to_phys = run_greedy(seed)

    lq_to_phys = best_lq_to_phys
    placed_lq_list = [lq for lq in logical_qubits if lq in lq_to_phys]
    n = len(placed_lq_list)

    # ── Initial 2-opt local search ────────────────────────────────────────
    improved = True
    max_passes = 10
    pass_count = 0
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        for i in range(n):
            for j in range(i + 1, n):
                lq1 = placed_lq_list[i]
                lq2 = placed_lq_list[j]
                if swap_delta(lq_to_phys, lq1, lq2) < -1e-9:
                    lq_to_phys[lq1], lq_to_phys[lq2] = lq_to_phys[lq2], lq_to_phys[lq1]
                    improved = True

    # ── Simulated Annealing to escape local optima ────────────────────────
    if n >= 4:
        current_cost = total_cost(lq_to_phys)

        # Scale temperature with problem: small positive floor avoids division by zero
        T = max(current_cost * 0.05 / max(n, 1), 0.5)
        T_min = 1e-4
        sa_iters = min(150 * n * n, 8000)
        # Geometric cooling schedule
        alpha = (T_min / T) ** (1.0 / sa_iters) if sa_iters > 0 else 1.0

        best_sa_cost = current_cost
        best_sa_mapping = dict(lq_to_phys)

        for _ in range(sa_iters):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while j == i:
                j = random.randint(0, n - 1)

            lq1 = placed_lq_list[i]
            lq2 = placed_lq_list[j]
            delta = swap_delta(lq_to_phys, lq1, lq2)

            if delta < 0 or (T > T_min and random.random() < math.exp(-max(delta / T, -500))):
                lq_to_phys[lq1], lq_to_phys[lq2] = lq_to_phys[lq2], lq_to_phys[lq1]
                current_cost += delta
                if current_cost < best_sa_cost:
                    best_sa_cost = current_cost
                    best_sa_mapping = dict(lq_to_phys)

            T *= alpha

        # Restore best solution found during SA
        lq_to_phys = best_sa_mapping

        # Final 2-opt cleanup after SA
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    lq1 = placed_lq_list[i]
                    lq2 = placed_lq_list[j]
                    if swap_delta(lq_to_phys, lq1, lq2) < -1e-9:
                        lq_to_phys[lq1], lq_to_phys[lq2] = lq_to_phys[lq2], lq_to_phys[lq1]
                        improved = True

    # ── Place remaining logical qubits ─────────────────────────────────────
    placed_phys_final = set(lq_to_phys.values())
    for lq in logical_qubits:
        if lq not in lq_to_phys:
            for p in physical_qubits:
                if p not in placed_phys_final:
                    lq_to_phys[lq] = p
                    placed_phys_final.add(p)
                    break

    # ── Build strict 1-to-1 bijection via in-place swap ───────────────────
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)