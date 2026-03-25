def init_mapping(self):
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import math
    import random

    gates_list = list(self.access.items())

    logical_qubit_set = set()
    for _, qubits in gates_list:
        for q in qubits:
            logical_qubit_set.add(q)

    # --- Step 1: Forward pass (earliest start/finish) ---
    qubit_ready_fwd = {}
    gate_earliest_start = []
    gate_earliest_finish = []
    for _, qubits in gates_list:
        es = max((qubit_ready_fwd.get(q, 0) for q in qubits), default=0)
        ef = es + 1
        gate_earliest_start.append(es)
        gate_earliest_finish.append(ef)
        for q in qubits:
            qubit_ready_fwd[q] = ef

    total_depth = max(gate_earliest_finish, default=1)
    max_layer = total_depth - 1

    # --- Step 2: Backward pass (critical path / slack) ---
    qubit_next_ls = defaultdict(lambda: total_depth)
    gate_latest_start = [0] * len(gates_list)
    for idx in range(len(gates_list) - 1, -1, -1):
        _, qubits = gates_list[idx]
        lf = min(qubit_next_ls[q] for q in qubits)
        ls = lf - 1
        gate_latest_start[idx] = ls
        for q in qubits:
            qubit_next_ls[q] = ls

    gate_slack = [gate_latest_start[i] - gate_earliest_start[i]
                  for i in range(len(gates_list))]
    max_slack = max(gate_slack, default=1) if gate_slack else 1

    if max_slack == 0:
        criticality = [2.5] * len(gates_list)
    else:
        criticality = [1.0 + 2.5 * (1.0 - s / max_slack) for s in gate_slack]

    # --- Step 3: First-order interaction weights with layer decay + criticality ---
    interaction_weight = defaultdict(float)
    half_life = max(max_layer / 3.5, 5.0)
    critical_end = max(1, int(max_layer * 0.15))
    crit_half_life = max(critical_end / 2.0, 1.0)

    for idx, (_, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_earliest_start[idx]
            w = math.exp(-layer * math.log(2) / half_life) + 0.08
            if layer <= critical_end:
                w += 1.5 * math.exp(-layer * math.log(2) / crit_half_life)
            w *= criticality[idx]
            interaction_weight[key] += w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_lq = len(logical_qubits)
    n_pq = len(physical_qubits)
    pq_set = set(physical_qubits)

    lq_interactions = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        lq_interactions[q1][q2] = w
        lq_interactions[q2][q1] = w

    # --- Step 3b: Second-order (transitive) interaction weights ---
    # If A-B and B-C interact, A and C benefit from proximity too.
    # Captures "chain" and "star" topologies in the logical interaction graph.
    second_order = defaultdict(float)
    alpha2 = 0.12
    for mid in logical_qubits:
        neighbors = list(lq_interactions[mid].items())
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                nb1, w1 = neighbors[i]
                nb2, w2 = neighbors[j]
                key = (min(nb1, nb2), max(nb1, nb2))
                second_order[key] += alpha2 * math.sqrt(w1 * w2)

    # Combined weights: first-order + transitive second-order
    combined_weight = defaultdict(float)
    for k, v in interaction_weight.items():
        combined_weight[k] += v
    for k, v in second_order.items():
        combined_weight[k] += v

    lq_combined = defaultdict(dict)
    for (q1, q2), w in combined_weight.items():
        lq_combined[q1][q2] = w
        lq_combined[q2][q1] = w

    pq_adj = {pq: [nb for nb in self.backend.get(pq, []) if nb in pq_set]
              for pq in physical_qubits}
    pq_degree = {pq: len(pq_adj[pq]) for pq in physical_qubits}

    # --- Step 4: Physical topology analysis ---
    pq_centrality = {}
    for pq in physical_qubits:
        c = sum(
            1.0 / self.distance_matrix[pq][other]
            for other in physical_qubits
            if other != pq and self.distance_matrix[pq][other] not in (0, float('inf'))
        )
        pq_centrality[pq] = c

    pq_r2 = {}
    for pq in physical_qubits:
        nbrs = set(pq_adj[pq])
        for nb in pq_adj[pq]:
            nbrs.update(pq_adj[nb])
        nbrs.discard(pq)
        pq_r2[pq] = list(nbrs)

    top_central = sorted(physical_qubits, key=lambda p: pq_centrality[p], reverse=True)
    top_degree_pq = sorted(physical_qubits, key=lambda p: pq_degree[p], reverse=True)

    # --- Step 5: Rearrangement-inequality Hungarian cost matrix (combined weights) ---
    lq_sorted_weights = {
        lq: sorted(lq_combined[lq].values(), reverse=True)
        for lq in logical_qubits
    }
    pq_sorted_dists = {
        pq: sorted(
            self.distance_matrix[pq][other]
            for other in physical_qubits
            if other != pq and self.distance_matrix[pq][other] != float('inf')
        )
        for pq in physical_qubits
    }

    cost_matrix = np.zeros((n_lq, n_pq))
    for i, lq in enumerate(logical_qubits):
        weights = lq_sorted_weights[lq]
        if not weights:
            continue
        for j, pq in enumerate(physical_qubits):
            dists = pq_sorted_dists[pq]
            cost = 0.0
            for k, w in enumerate(weights):
                d = dists[k] if k < len(dists) else (dists[-1] + k - len(dists) + 1)
                cost += w * d
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    hungarian_map = {
        logical_qubits[r]: physical_qubits[c]
        for r, c in zip(row_ind, col_ind)
    }

    # --- Step 6: Greedy BFS seed (using combined weights) ---
    def greedy_bfs_seed(anchor_pq=None, seed_pair=None):
        lq_phys = {}
        phys_used = set()

        if seed_pair is not None:
            slq1, slq2 = seed_pair
        elif combined_weight:
            slq1, slq2 = max(combined_weight, key=combined_weight.__getitem__)
        else:
            slq1 = logical_qubits[0]
            slq2 = logical_qubits[1] if n_lq > 1 else logical_qubits[0]

        if anchor_pq is not None and anchor_pq in pq_adj:
            sp1 = anchor_pq
            sp2 = max(pq_adj[sp1], key=lambda p: pq_degree[p], default=sp1)
        else:
            best_score, sp1, sp2 = -1, physical_qubits[0], physical_qubits[min(1, n_pq - 1)]
            for p1 in physical_qubits:
                for p2 in pq_adj[p1]:
                    s = pq_degree[p1] + pq_degree[p2]
                    if s > best_score:
                        best_score, sp1, sp2 = s, p1, p2

        w1 = sum(lq_combined[slq1].values())
        w2 = sum(lq_combined[slq2].values())
        if (w1 >= w2) == (pq_degree[sp1] >= pq_degree[sp2]):
            lq_phys[slq1], lq_phys[slq2] = sp1, sp2
        else:
            lq_phys[slq1], lq_phys[slq2] = sp2, sp1

        phys_used.update(lq_phys.values())
        placed = set(lq_phys.keys())

        while len(placed) < n_lq:
            best_lq, best_score_lq = None, -1
            for lq in logical_qubits:
                if lq in placed:
                    continue
                s = sum(lq_combined[lq].get(p, 0) for p in placed)
                if s > best_score_lq:
                    best_score_lq, best_lq = s, lq
            if best_lq is None:
                best_lq = next(lq for lq in logical_qubits if lq not in placed)

            best_pq, best_pq_cost = None, float('inf')
            for pq in physical_qubits:
                if pq in phys_used:
                    continue
                cost = sum(
                    w * self.distance_matrix[pq][lq_phys[nb]]
                    for nb, w in lq_combined[best_lq].items()
                    if nb in placed
                )
                if cost < best_pq_cost:
                    best_pq_cost, best_pq = cost, pq
            if best_pq is None:
                best_pq = next(pq for pq in physical_qubits if pq not in phys_used)

            lq_phys[best_lq] = best_pq
            phys_used.add(best_pq)
            placed.add(best_lq)

        return lq_phys

    # --- Step 7: Build bijection ---
    def make_mapping(lq_phys):
        md = list(range(self.num_qubits))
        for lq, pq in lq_phys.items():
            md[lq] = pq
        assigned = set(lq_phys.values())
        remaining = [pq for pq in range(self.num_qubits) if pq not in assigned]
        ri = 0
        for lq in range(self.num_qubits):
            if lq not in lq_phys:
                md[lq] = remaining[ri]
                ri += 1
        rmd = list(range(self.num_qubits))
        for lq in range(self.num_qubits):
            rmd[md[lq]] = lq
        return md, rmd

    # --- Step 8: Cost utilities (evaluation on first-order; optimization on combined) ---
    def qap_cost(md):
        return sum(
            w * self.distance_matrix[md[q1]][md[q2]]
            for (q1, q2), w in interaction_weight.items()
        )

    def swap_delta(p1, p2, md, rmd):
        # Uses combined weights for richer optimization signal
        a, b = rmd[p1], rmd[p2]
        delta = 0.0
        for nb, w in lq_combined[a].items():
            pp = md[nb]
            if pp == p2:
                continue
            delta += w * (self.distance_matrix[p2][pp] - self.distance_matrix[p1][pp])
        for nb, w in lq_combined[b].items():
            pp = md[nb]
            if pp == p1:
                continue
            delta += w * (self.distance_matrix[p1][pp] - self.distance_matrix[p2][pp])
        return delta

    def do_swap(p1, p2, md, rmd):
        a, b = rmd[p1], rmd[p2]
        md[a], md[b] = p2, p1
        rmd[p1], rmd[p2] = b, a

    lq_by_degree = sorted(
        logical_qubits,
        key=lambda lq: sum(lq_combined[lq].values()),
        reverse=True
    )

    # --- Step 9: Two-tier local search (fast: r2 neighbors; periodic: full sweep) ---
    def local_search(md, rmd, max_iters=400, full_every=4):
        for iteration in range(max_iters):
            improved = False
            do_full = (iteration % full_every == 0)
            for lq in lq_by_degree:
                p1 = md[lq]
                best_d, best_p2 = -1e-9, -1
                for p2 in pq_r2[p1]:
                    d = swap_delta(p1, p2, md, rmd)
                    if d < best_d:
                        best_d, best_p2 = d, p2
                if do_full:
                    r2_set = set(pq_r2[p1])
                    r2_set.add(p1)
                    for p2 in physical_qubits:
                        if p2 in r2_set:
                            continue
                        d = swap_delta(p1, p2, md, rmd)
                        if d < best_d:
                            best_d, best_p2 = d, p2
                if best_p2 != -1:
                    do_swap(p1, best_p2, md, rmd)
                    improved = True
            if not improved:
                break
        return md, rmd

    def local_search_full(md, rmd, max_iters=400):
        for _ in range(max_iters):
            improved = False
            for lq in lq_by_degree:
                p1 = md[lq]
                best_d, best_p2 = -1e-9, -1
                for p2 in physical_qubits:
                    if p2 == p1:
                        continue
                    d = swap_delta(p1, p2, md, rmd)
                    if d < best_d:
                        best_d, best_p2 = d, p2
                if best_p2 != -1:
                    do_swap(p1, best_p2, md, rmd)
                    improved = True
            if not improved:
                break
        return md, rmd

    # --- Step 10: Multi-start seeds ---
    sorted_combined = sorted(combined_weight.items(), key=lambda x: x[1], reverse=True)
    alt_pairs = [pair for pair, _ in sorted_combined[1:3]] if len(sorted_combined) > 1 else []

    seed_configs = [
        (hungarian_map, None),
        (None, (None, None)),
    ]
    for anchor in top_central[:2]:
        seed_configs.append((None, (anchor, None)))
    for anchor in top_degree_pq[:2]:
        seed_configs.append((None, (anchor, None)))
    for pair in alt_pairs:
        seed_configs.append((None, (None, pair)))

    best_md, best_rmd, best_cost = None, None, float('inf')
    population = []  # top solutions for crossover diversity

    for fixed_map, bfs_args in seed_configs:
        if fixed_map is not None:
            md, rmd = make_mapping(fixed_map)
        else:
            anchor_pq, seed_pair = bfs_args
            bfs_map = greedy_bfs_seed(anchor_pq=anchor_pq, seed_pair=seed_pair)
            md, rmd = make_mapping(bfs_map)
        md, rmd = local_search_full(md, rmd)
        c = qap_cost(md)
        population.append((c, md[:], rmd[:]))
        if c < best_cost:
            best_cost = c
            best_md, best_rmd = md[:], rmd[:]

    population = sorted(population, key=lambda x: x[0])[:5]

    # --- Step 11: ILS with five diversified perturbation strategies ---
    rng = random.Random(42)
    n_restarts = min(32, max(8, n_lq // 2))

    for restart in range(n_restarts):
        # Cycle through population as ILS base (not just single best)
        base_idx = restart % len(population)
        _, base_md, base_rmd = population[base_idx]
        md = base_md[:]
        rmd = base_rmd[:]
        strategy = restart % 5

        if strategy == 0:
            # Cyclic rotation of random subset
            k = rng.randint(max(3, n_lq // 4), max(4, n_lq // 2 + 1))
            k = min(k, n_lq)
            sample = rng.sample(logical_qubits, k)
            positions = [md[lq] for lq in sample]
            offset = rng.randint(1, k - 1)
            rotated = positions[offset:] + positions[:offset]
            for lq, tgt in zip(sample, rotated):
                cur = md[lq]
                if cur != tgt:
                    do_swap(cur, tgt, md, rmd)

        elif strategy == 1:
            # Random pairwise swaps
            n_swaps = rng.randint(2, max(3, n_lq // 3))
            pool = rng.sample(logical_qubits, min(n_swaps * 2, n_lq))
            for i in range(0, len(pool) - 1, 2):
                do_swap(md[pool[i]], md[pool[i + 1]], md, rmd)

        elif strategy == 2:
            # Targeted worst-pair relocation: jointly move bad pair to adjacent physical qubits
            worst = sorted(
                interaction_weight.items(),
                key=lambda x: x[1] * self.distance_matrix[md[x[0][0]]][md[x[0][1]]],
                reverse=True
            )
            moved = set()
            for (q1, q2), _ in worst[:max(2, n_lq // 5)]:
                if q1 in moved or q2 in moved:
                    continue
                # Find best adjacent physical pair for (q1, q2)
                best_cost_pair = float('inf')
                best_pp1, best_pp2 = md[q1], md[q2]
                for pp1 in top_central[:max(6, n_pq // 4)]:
                    for pp2 in pq_adj[pp1]:
                        cost_pair = 0.0
                        for nb, w in lq_combined[q1].items():
                            pnb = pp2 if nb == q2 else md[nb]
                            cost_pair += w * self.distance_matrix[pp1][pnb]
                        for nb, w in lq_combined[q2].items():
                            pnb = pp1 if nb == q1 else md[nb]
                            cost_pair += w * self.distance_matrix[pp2][pnb]
                        if cost_pair < best_cost_pair:
                            best_cost_pair = cost_pair
                            best_pp1, best_pp2 = pp1, pp2
                cur_p1 = md[q1]
                if best_pp1 != cur_p1:
                    do_swap(cur_p1, best_pp1, md, rmd)
                cur_p2 = md[q2]
                if best_pp2 != cur_p2:
                    do_swap(cur_p2, best_pp2, md, rmd)
                moved.update([q1, q2])

        elif strategy == 3:
            # Subgraph re-embedding in a new physical neighborhood
            seed_lq = rng.choice(lq_by_degree[:max(1, n_lq // 4)])
            sub_size = rng.randint(2, max(3, n_lq // 5))
            subgraph = [seed_lq]
            lq_frontier = sorted(
                lq_combined[seed_lq].keys(),
                key=lambda q: lq_combined[seed_lq][q],
                reverse=True
            )
            for lq in lq_frontier:
                if len(subgraph) >= sub_size:
                    break
                if lq not in subgraph:
                    subgraph.append(lq)

            cand_anchors = top_central[:max(3, n_pq // 6)]
            new_anchor = rng.choice(cand_anchors)
            phys_blocked = set(md[lq] for lq in logical_qubits if lq not in subgraph)
            new_positions = []
            phys_frontier_list = [new_anchor]
            visited_pq = {new_anchor}
            for pq in phys_frontier_list:
                if pq not in phys_blocked:
                    new_positions.append(pq)
                if len(new_positions) >= len(subgraph):
                    break
                for nb in pq_adj[pq]:
                    if nb not in visited_pq:
                        visited_pq.add(nb)
                        phys_frontier_list.append(nb)

            if len(new_positions) < len(subgraph):
                for pq in physical_qubits:
                    if pq not in phys_blocked and pq not in set(new_positions):
                        new_positions.append(pq)
                        if len(new_positions) >= len(subgraph):
                            break

            for i, lq in enumerate(subgraph):
                if i < len(new_positions):
                    tgt = new_positions[i]
                    cur = md[lq]
                    if cur != tgt:
                        do_swap(cur, tgt, md, rmd)

        else:
            # Strategy 4: population crossover — inherit positions of top interacting
            # qubits from a different elite solution, creating a hybrid start point
            if len(population) >= 2:
                other_idx = rng.randint(0, len(population) - 1)
                _, other_md, _ = population[other_idx]
                crossover_size = max(1, n_lq // 3)
                crossover_lqs = lq_by_degree[:crossover_size]
                for lq in crossover_lqs:
                    target_pq = other_md[lq]
                    cur_pq = md[lq]
                    if cur_pq != target_pq:
                        do_swap(cur_pq, target_pq, md, rmd)

        md, rmd = local_search(md, rmd)
        c = qap_cost(md)
        if c < best_cost:
            best_cost = c
            best_md = md[:]
            best_rmd = rmd[:]
        # Maintain elite population for crossover diversity
        if len(population) < 5 or c < population[-1][0]:
            population.append((c, md[:], rmd[:]))
            population = sorted(population, key=lambda x: x[0])[:5]

    # --- Step 12: SA refinement with reheating to escape deeper local optima ---
    md = best_md[:]
    rmd = best_rmd[:]
    current_cost = best_cost

    active_pqs = list(set(md[lq] for lq in logical_qubits if lq_combined[lq]))
    if len(active_pqs) < 2:
        active_pqs = physical_qubits

    n_sa_iters = max(4000, n_lq * 350)
    T_start = max(current_cost * 0.05, 0.5)
    T_end = max(current_cost * 0.00004, 1e-4)
    alpha = (T_end / T_start) ** (1.0 / n_sa_iters)
    T = T_start

    no_improve_streak = 0
    reheat_interval = n_sa_iters // 6

    for _ in range(n_sa_iters):
        p1, p2 = rng.sample(active_pqs, 2)
        delta = swap_delta(p1, p2, md, rmd)
        if delta < 0 or (T > 1e-9 and rng.random() < math.exp(-delta / T)):
            do_swap(p1, p2, md, rmd)
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_md = md[:]
                best_rmd = rmd[:]
                no_improve_streak = 0
            else:
                no_improve_streak += 1
        else:
            no_improve_streak += 1

        T *= alpha

        # Reheating: if thermodynamically stuck, boost T to allow fresh exploration
        if no_improve_streak >= reheat_interval:
            T = max(T, T_start * 0.08)
            no_improve_streak = 0

    # --- Step 13: Final full local search on best solution ---
    best_md, best_rmd = local_search_full(best_md, best_rmd, max_iters=500)

    self.mapping_dict = best_md
    self.reverse_mapping_dict = best_rmd

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)