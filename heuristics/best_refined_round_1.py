def init_mapping(self):
    """
    DAG-Aware K-Core Refined Placement

    Combines k-core decomposition ordering with DAG-aware temporal decay
    weighting, critical path analysis, second-order transitive interactions,
    Hungarian assignment seeding, and efficient pairwise local search with
    lightweight ILS perturbations.
    """
    from collections import defaultdict, deque
    import math
    import random

    rng = random.Random(42)

    # ================================================================== #
    # 1. Build DAG and compute gate layers + critical path               #
    # ================================================================== #
    all_gates = sorted(self.access.keys())
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]

    logical_qubit_set = set()
    for qubits in self.access.values():
        logical_qubit_set.update(qubits)
    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits or not two_qubit_gates:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_lq = len(logical_qubits)
    n_pq = len(physical_qubits)

    # Build DAG via last-gate-on-qubit tracking
    last_gate_on_qubit = {}
    successors_dag = defaultdict(set)
    predecessors_dag = defaultdict(set)
    for g in all_gates:
        for q in self.access[g]:
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                successors_dag[pred].add(g)
                predecessors_dag[g].add(pred)
            last_gate_on_qubit[q] = g

    # Gate layer via topological BFS (Kahn's algorithm)
    in_degree = {g: len(predecessors_dag[g]) for g in all_gates}
    gate_layer = {g: 0 for g in all_gates}
    queue = deque(g for g in all_gates if in_degree[g] == 0)
    topo_order = []
    temp_in = dict(in_degree)
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in successors_dag[g]:
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)

    # Critical path: remaining depth from each gate
    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors_dag[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    max_cp = max((critical_path[g] for g in two_qubit_gates), default=1)

    # ================================================================== #
    # 2. Build temporally-decayed, criticality-weighted interactions      #
    # ================================================================== #
    alpha_decay = math.log(10.0) / (max_layer + 1)
    interaction_weight = defaultdict(float)
    interaction_neighbors = defaultdict(dict)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        layer = gate_layer[g]
        cp = critical_path[g]
        # Temporal decay: early gates weighted exponentially more
        w = math.exp(-alpha_decay * layer)
        # Critical path boost: gates on critical path get extra weight
        w *= (1.0 + 1.5 * cp / max_cp) if max_cp > 0 else 1.0
        # Minimum floor to not ignore late gates entirely
        w = max(w, 0.05)
        key = (min(q1, q2), max(q1, q2))
        interaction_weight[key] += w
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0.0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0.0) + w

    # Second-order (transitive) interactions
    alpha2 = 0.10
    second_order = defaultdict(float)
    for mid in logical_qubits:
        neighbors = list(interaction_neighbors[mid].items())
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                nb1, w1 = neighbors[i]
                nb2, w2 = neighbors[j]
                key = (min(nb1, nb2), max(nb1, nb2))
                second_order[key] += alpha2 * math.sqrt(w1 * w2)

    # Combined weights
    combined_weight = defaultdict(float)
    for k, v in interaction_weight.items():
        combined_weight[k] += v
    for k, v in second_order.items():
        combined_weight[k] += v

    lq_combined = defaultdict(dict)
    for (q1, q2), w in combined_weight.items():
        lq_combined[q1][q2] = w
        lq_combined[q2][q1] = w

    weighted_degree = {q: sum(lq_combined[q].values()) for q in logical_qubits}

    # ================================================================== #
    # 3. Hardware topology analysis                                       #
    # ================================================================== #
    pq_adj = {pq: [nb for nb in self.backend.get(pq, []) if nb in set(physical_qubits)]
              for pq in physical_qubits}
    pq_degree = {pq: len(pq_adj[pq]) for pq in physical_qubits}

    # Harmonic centrality (more robust than sum-of-distances)
    pq_centrality = {}
    for pq in physical_qubits:
        c = sum(
            1.0 / self.distance_matrix[pq][other]
            for other in physical_qubits
            if other != pq and self.distance_matrix[pq][other] not in (0, float('inf'))
        )
        pq_centrality[pq] = c

    # Radius-2 neighborhoods for efficient local search
    pq_r2 = {}
    for pq in physical_qubits:
        nbrs = set(pq_adj[pq])
        for nb in pq_adj[pq]:
            nbrs.update(pq_adj[nb])
        nbrs.discard(pq)
        pq_r2[pq] = list(nbrs)

    phys_by_centrality = sorted(physical_qubits, key=lambda p: pq_centrality[p], reverse=True)

    # ================================================================== #
    # 4. K-core decomposition for placement ordering                      #
    # ================================================================== #
    adj = defaultdict(set)
    for q in logical_qubits:
        for nb in interaction_neighbors[q]:
            adj[q].add(nb)

    degree = {q: len(adj[q]) for q in logical_qubits}
    remaining = set(logical_qubits)
    peeling_order = []

    while remaining:
        min_q = min(remaining, key=lambda q: degree[q])
        peeling_order.append(min_q)
        remaining.remove(min_q)
        for nb in adj[min_q]:
            if nb in remaining:
                degree[nb] -= 1

    # Reverse: most interconnected first
    kcore_placement_order = list(reversed(peeling_order))

    # ================================================================== #
    # 5. Greedy BFS placement using combined weights                      #
    # ================================================================== #
    def greedy_bfs_seed(placement_order, anchor_pq=None):
        lq_phys = {}
        used_phys = set()

        for lq in placement_order:
            neighbors_placed = [
                nb for nb in lq_combined[lq] if nb in lq_phys
            ]

            if not neighbors_placed:
                if anchor_pq is not None and anchor_pq not in used_phys:
                    best_phys = anchor_pq
                    anchor_pq = None
                else:
                    best_phys = None
                    for p in phys_by_centrality:
                        if p not in used_phys:
                            best_phys = p
                            break
                    if best_phys is None:
                        best_phys = next(p for p in physical_qubits if p not in used_phys)
            else:
                best_phys = None
                best_score = float('inf')
                unplaced_nb_count = sum(1 for nb in lq_combined[lq] if nb not in lq_phys)
                for p in physical_qubits:
                    if p in used_phys:
                        continue
                    dist_cost = sum(
                        lq_combined[lq][nb] * self.distance_matrix[p][lq_phys[nb]]
                        for nb in neighbors_placed
                    )
                    free_nb = sum(1 for adj_p in pq_adj[p] if adj_p not in used_phys)
                    capacity_penalty = max(0, unplaced_nb_count - free_nb) * 0.15
                    score = dist_cost + capacity_penalty
                    if score < best_score:
                        best_score = score
                        best_phys = p

            lq_phys[lq] = best_phys
            used_phys.add(best_phys)

        return lq_phys

    # ================================================================== #
    # 6. Hungarian assignment seed (rearrangement inequality)             #
    # ================================================================== #
    def hungarian_seed():
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return None

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
        return {
            logical_qubits[r]: physical_qubits[c]
            for r, c in zip(row_ind, col_ind)
        }

    # ================================================================== #
    # 7. Bijection builder + cost utilities                               #
    # ================================================================== #
    def make_mapping(lq_phys):
        md = list(range(self.num_qubits))
        rmd = list(range(self.num_qubits))
        for lq, pq in lq_phys.items():
            cur = md[lq]
            if cur == pq:
                continue
            displaced = rmd[pq]
            md[lq] = pq
            md[displaced] = cur
            rmd[pq] = lq
            rmd[cur] = displaced
        return md, rmd

    def qap_cost(md):
        return sum(
            w * self.distance_matrix[md[q1]][md[q2]]
            for (q1, q2), w in interaction_weight.items()
        )

    def swap_delta(p1, p2, md, rmd):
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

    # ================================================================== #
    # 8. Two-tier local search                                            #
    # ================================================================== #
    lq_by_degree = sorted(
        logical_qubits,
        key=lambda lq: weighted_degree.get(lq, 0),
        reverse=True
    )

    def local_search(md, rmd, max_iters=300, full_every=5):
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

    # ================================================================== #
    # 9. Multi-start seeds                                                #
    # ================================================================== #
    best_md, best_rmd, best_cost = None, None, float('inf')
    population = []

    # Seed 1: K-core order, default centrality anchor
    kcore_map = greedy_bfs_seed(kcore_placement_order)
    md, rmd = make_mapping(kcore_map)
    md, rmd = local_search_full(md, rmd)
    c = qap_cost(md)
    population.append((c, md[:], rmd[:]))
    if c < best_cost:
        best_cost, best_md, best_rmd = c, md[:], rmd[:]

    # Seed 2: K-core order, different anchor points
    for anchor in phys_by_centrality[1:4]:
        kcore_map2 = greedy_bfs_seed(kcore_placement_order, anchor_pq=anchor)
        md, rmd = make_mapping(kcore_map2)
        md, rmd = local_search_full(md, rmd)
        c = qap_cost(md)
        population.append((c, md[:], rmd[:]))
        if c < best_cost:
            best_cost, best_md, best_rmd = c, md[:], rmd[:]

    # Seed 3: Weighted-degree order (highest interaction weight first)
    wd_order = sorted(logical_qubits, key=lambda q: weighted_degree.get(q, 0), reverse=True)
    wd_map = greedy_bfs_seed(wd_order)
    md, rmd = make_mapping(wd_map)
    md, rmd = local_search_full(md, rmd)
    c = qap_cost(md)
    population.append((c, md[:], rmd[:]))
    if c < best_cost:
        best_cost, best_md, best_rmd = c, md[:], rmd[:]

    # Seed 4: Hungarian assignment
    hung_map = hungarian_seed()
    if hung_map is not None:
        md, rmd = make_mapping(hung_map)
        md, rmd = local_search_full(md, rmd)
        c = qap_cost(md)
        population.append((c, md[:], rmd[:]))
        if c < best_cost:
            best_cost, best_md, best_rmd = c, md[:], rmd[:]

    # Seed 5: Strongest-pair seed with k-core fill
    if combined_weight:
        top_pair = max(combined_weight, key=combined_weight.__getitem__)
        sq1, sq2 = top_pair
        best_pair_score = float('inf')
        best_pp1, best_pp2 = physical_qubits[0], physical_qubits[min(1, n_pq - 1)]
        for p1 in phys_by_centrality[:20]:
            for p2 in pq_adj[p1]:
                s = (1.0 / (pq_centrality[p1] + 0.01)) + (1.0 / (pq_centrality[p2] + 0.01))
                if s < best_pair_score:
                    best_pair_score = s
                    best_pp1, best_pp2 = p1, p2

        for a, b in [(sq1, sq2), (sq2, sq1)]:
            partial = {a: best_pp1, b: best_pp2}
            remaining_order = [lq for lq in kcore_placement_order if lq not in partial]
            used = set(partial.values())
            for lq in remaining_order:
                neighbors_placed = [nb for nb in lq_combined[lq] if nb in partial]
                if not neighbors_placed:
                    for p in phys_by_centrality:
                        if p not in used:
                            partial[lq] = p
                            used.add(p)
                            break
                else:
                    best_p, best_s = None, float('inf')
                    for p in physical_qubits:
                        if p in used:
                            continue
                        s = sum(lq_combined[lq][nb] * self.distance_matrix[p][partial[nb]]
                                for nb in neighbors_placed)
                        if s < best_s:
                            best_s, best_p = s, p
                    partial[lq] = best_p
                    used.add(best_p)

            md, rmd = make_mapping(partial)
            md, rmd = local_search_full(md, rmd)
            c = qap_cost(md)
            population.append((c, md[:], rmd[:]))
            if c < best_cost:
                best_cost, best_md, best_rmd = c, md[:], rmd[:]

    population = sorted(population, key=lambda x: x[0])[:6]

    # ================================================================== #
    # 10. ILS with diversified perturbations                              #
    # ================================================================== #
    n_restarts = min(28, max(8, n_lq // 2))

    for restart in range(n_restarts):
        base_idx = restart % len(population)
        _, base_md, base_rmd = population[base_idx]
        md = base_md[:]
        rmd = base_rmd[:]
        strategy = restart % 4

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
            # Worst-pair relocation
            worst = sorted(
                interaction_weight.items(),
                key=lambda x: x[1] * self.distance_matrix[md[x[0][0]]][md[x[0][1]]],
                reverse=True
            )
            moved = set()
            for (q1, q2), _ in worst[:max(2, n_lq // 5)]:
                if q1 in moved or q2 in moved:
                    continue
                best_cost_pair = float('inf')
                best_pp1, best_pp2 = md[q1], md[q2]
                for pp1 in phys_by_centrality[:max(8, n_pq // 4)]:
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

        else:
            # Population crossover
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
            best_cost, best_md, best_rmd = c, md[:], rmd[:]
        if len(population) < 6 or c < population[-1][0]:
            population.append((c, md[:], rmd[:]))
            population = sorted(population, key=lambda x: x[0])[:6]

    # ================================================================== #
    # 11. Simulated annealing refinement                                  #
    # ================================================================== #
    md = best_md[:]
    rmd = best_rmd[:]
    current_cost = best_cost

    active_pqs = list(set(md[lq] for lq in logical_qubits if lq_combined[lq]))
    if len(active_pqs) < 2:
        active_pqs = physical_qubits

    n_sa_iters = max(5000, n_lq * 400)
    T_start = max(current_cost * 0.05, 0.5)
    T_end = max(current_cost * 0.00003, 1e-4)
    sa_alpha = (T_end / T_start) ** (1.0 / n_sa_iters) if n_sa_iters > 0 else 1.0
    T = T_start

    no_improve = 0
    reheat_interval = n_sa_iters // 5

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
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        T *= sa_alpha
        if no_improve >= reheat_interval:
            T = max(T, T_start * 0.1)
            no_improve = 0

    # ================================================================== #
    # 12. Final full local search                                         #
    # ================================================================== #
    best_md, best_rmd = local_search_full(best_md, best_rmd, max_iters=500)

    self.mapping_dict = best_md
    self.reverse_mapping_dict = best_rmd

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)