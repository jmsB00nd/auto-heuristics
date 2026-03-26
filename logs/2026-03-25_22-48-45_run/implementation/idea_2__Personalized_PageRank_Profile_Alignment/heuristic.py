def init_mapping(self):
    """
    Personalized PageRank Profile Alignment.

    For each logical qubit, compute a Personalized PageRank (PPR) vector over the
    logical interaction graph. Similarly compute PPR vectors for each physical qubit
    on the hardware graph. Match logical-to-physical by minimizing L2 divergence
    between sorted PPR profiles via the Hungarian algorithm, then refine with local search.
    """
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import math
    import random

    gates_list = list(self.access.items())

    # Collect logical qubits that appear in the circuit
    logical_qubit_set = set()
    for _, qubits in gates_list:
        for q in qubits:
            logical_qubit_set.add(q)

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

    # --- Step 1: Build weighted logical interaction graph ---
    qubit_ready = {}
    gate_layers = []
    for _, qubits in gates_list:
        es = max((qubit_ready.get(q, 0) for q in qubits), default=0)
        ef = es + 1
        gate_layers.append(es)
        for q in qubits:
            qubit_ready[q] = ef

    total_depth = max((qubit_ready.get(q, 0) for q in logical_qubit_set), default=1)
    max_layer = max(total_depth - 1, 1)
    half_life = max(max_layer / 4.0, 4.0)

    interaction_weight = defaultdict(float)
    for idx, (_, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_layers[idx]
            w = math.exp(-layer * math.log(2) / half_life) + 0.05
            interaction_weight[key] += w

    lq_index = {lq: i for i, lq in enumerate(logical_qubits)}
    pq_index = {pq: i for i, pq in enumerate(physical_qubits)}

    lq_adj = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        lq_adj[q1][q2] = w
        lq_adj[q2][q1] = w

    # --- Step 2: Compute PPR vectors for logical qubits ---
    alpha = 0.85

    W_logical = np.zeros((n_lq, n_lq))
    for (q1, q2), w in interaction_weight.items():
        i, j = lq_index[q1], lq_index[q2]
        W_logical[i, j] = w
        W_logical[j, i] = w

    row_sums = W_logical.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    T_logical = W_logical / row_sums[:, np.newaxis]

    ppr_logical = np.zeros((n_lq, n_lq))
    n_iters_ppr = 40

    for seed_idx in range(n_lq):
        ppr = np.zeros(n_lq)
        ppr[seed_idx] = 1.0
        for _ in range(n_iters_ppr):
            ppr = (1.0 - alpha) * np.eye(n_lq)[seed_idx] + alpha * (T_logical.T @ ppr)
        norm = np.linalg.norm(ppr)
        if norm > 1e-12:
            ppr /= norm
        ppr_logical[seed_idx] = ppr

    # --- Step 3: Compute PPR vectors for physical qubits ---
    W_hardware = np.zeros((n_pq, n_pq))
    for pq in physical_qubits:
        for nb in self.backend.get(pq, []):
            if nb in pq_index:
                i, j = pq_index[pq], pq_index[nb]
                W_hardware[i, j] = 1.0

    row_sums_hw = W_hardware.sum(axis=1)
    row_sums_hw[row_sums_hw == 0] = 1.0
    T_hardware = W_hardware / row_sums_hw[:, np.newaxis]

    ppr_hardware = np.zeros((n_pq, n_pq))
    for seed_idx in range(n_pq):
        ppr = np.zeros(n_pq)
        ppr[seed_idx] = 1.0
        for _ in range(n_iters_ppr):
            ppr = (1.0 - alpha) * np.eye(n_pq)[seed_idx] + alpha * (T_hardware.T @ ppr)
        norm = np.linalg.norm(ppr)
        if norm > 1e-12:
            ppr /= norm
        ppr_hardware[seed_idx] = ppr

    # --- Step 4: Build cost matrix using sorted PPR profile L2 distance ---
    sorted_ppr_logical = np.zeros((n_lq, n_lq))
    for i in range(n_lq):
        sorted_ppr_logical[i] = np.sort(ppr_logical[i])[::-1]

    sorted_ppr_hardware = np.zeros((n_pq, n_pq))
    for i in range(n_pq):
        sorted_ppr_hardware[i] = np.sort(ppr_hardware[i])[::-1]

    dim = max(n_lq, n_pq)
    padded_logical = np.zeros((n_lq, dim))
    padded_hardware = np.zeros((n_pq, dim))
    padded_logical[:, :n_lq] = sorted_ppr_logical
    padded_hardware[:, :n_pq] = sorted_ppr_hardware

    cost_matrix = np.zeros((n_lq, n_pq))
    for i in range(n_lq):
        for j in range(n_pq):
            diff = padded_logical[i] - padded_hardware[j]
            cost_matrix[i, j] = np.dot(diff, diff)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    ppr_map = {
        logical_qubits[r]: physical_qubits[c]
        for r, c in zip(row_ind, col_ind)
    }

    # --- Step 5: Rearrangement-inequality seed for diversity ---
    lq_sorted_weights = {
        lq: sorted(lq_adj[lq].values(), reverse=True)
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

    rearr_cost = np.zeros((n_lq, n_pq))
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
            rearr_cost[i, j] = cost

    row_ind2, col_ind2 = linear_sum_assignment(rearr_cost)
    rearr_map = {
        logical_qubits[r]: physical_qubits[c]
        for r, c in zip(row_ind2, col_ind2)
    }

    # --- Step 6: Greedy BFS seed ---
    pq_adj = {pq: [nb for nb in self.backend.get(pq, []) if nb in pq_set]
              for pq in physical_qubits}
    pq_degree = {pq: len(pq_adj[pq]) for pq in physical_qubits}

    combined_weight = defaultdict(float)
    for k, v in interaction_weight.items():
        combined_weight[k] += v
    alpha2 = 0.12
    for mid in logical_qubits:
        neighbors = list(lq_adj[mid].items())
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                nb1, w1 = neighbors[i]
                nb2, w2 = neighbors[j]
                key = (min(nb1, nb2), max(nb1, nb2))
                combined_weight[key] += alpha2 * math.sqrt(w1 * w2)

    lq_combined = defaultdict(dict)
    for (q1, q2), w in combined_weight.items():
        lq_combined[q1][q2] = w
        lq_combined[q2][q1] = w

    def greedy_bfs_seed():
        lq_phys = {}
        phys_used = set()
        if combined_weight:
            slq1, slq2 = max(combined_weight, key=combined_weight.__getitem__)
        else:
            slq1 = logical_qubits[0]
            slq2 = logical_qubits[1] if n_lq > 1 else logical_qubits[0]

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

    # --- Utility functions ---
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

    lq_by_degree = sorted(
        logical_qubits,
        key=lambda lq: sum(lq_combined[lq].values()),
        reverse=True
    )

    pq_r2 = {}
    for pq in physical_qubits:
        nbrs = set(pq_adj[pq])
        for nb in pq_adj[pq]:
            nbrs.update(pq_adj[nb])
        nbrs.discard(pq)
        pq_r2[pq] = list(nbrs)

    def local_search(md, rmd, max_iters=400):
        for iteration in range(max_iters):
            improved = False
            do_full = (iteration % 4 == 0)
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

    def local_search_full(md, rmd, max_iters=500):
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

    # --- Step 7: Multi-start evaluation ---
    seeds = [
        ("ppr", ppr_map),
        ("rearrangement", rearr_map),
        ("bfs", greedy_bfs_seed()),
    ]

    best_md, best_rmd, best_cost = None, None, float('inf')
    population = []

    for name, seed_map in seeds:
        md, rmd = make_mapping(seed_map)
        md, rmd = local_search_full(md, rmd)
        c = qap_cost(md)
        population.append((c, md[:], rmd[:]))
        if c < best_cost:
            best_cost = c
            best_md, best_rmd = md[:], rmd[:]

    population = sorted(population, key=lambda x: x[0])[:5]

    # --- Step 8: ILS with perturbation ---
    rng = random.Random(42)
    n_restarts = min(30, max(8, n_lq // 2))

    for restart in range(n_restarts):
        base_idx = restart % len(population)
        _, base_md, base_rmd = population[base_idx]
        md = base_md[:]
        rmd = base_rmd[:]
        strategy = restart % 4

        if strategy == 0:
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
            n_swaps = rng.randint(2, max(3, n_lq // 3))
            pool = rng.sample(logical_qubits, min(n_swaps * 2, n_lq))
            for i in range(0, len(pool) - 1, 2):
                do_swap(md[pool[i]], md[pool[i + 1]], md, rmd)
        elif strategy == 2:
            worst = sorted(
                interaction_weight.items(),
                key=lambda x: x[1] * self.distance_matrix[md[x[0][0]]][md[x[0][1]]],
                reverse=True
            )
            moved = set()
            pq_centrality_sorted = sorted(
                physical_qubits,
                key=lambda p: sum(
                    1.0 / self.distance_matrix[p][o]
                    for o in physical_qubits
                    if o != p and self.distance_matrix[p][o] not in (0, float('inf'))
                ),
                reverse=True
            )
            for (q1, q2), _ in worst[:max(2, n_lq // 5)]:
                if q1 in moved or q2 in moved:
                    continue
                best_cost_pair = float('inf')
                best_pp1, best_pp2 = md[q1], md[q2]
                for pp1 in pq_centrality_sorted[:max(8, n_pq // 4)]:
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
            n_triples = rng.randint(1, max(2, n_lq // 6))
            candidates = lq_by_degree[:max(6, n_lq // 2)]
            for _ in range(n_triples):
                if len(candidates) < 3:
                    break
                triple = rng.sample(candidates, 3)
                p0, p1_t, p2_t = md[triple[0]], md[triple[1]], md[triple[2]]
                do_swap(p0, p1_t, md, rmd)
                do_swap(p1_t, p2_t, md, rmd)

        md, rmd = local_search(md, rmd)
        c = qap_cost(md)
        if c < best_cost:
            best_cost = c
            best_md = md[:]
            best_rmd = rmd[:]
        if len(population) < 5 or c < population[-1][0]:
            population.append((c, md[:], rmd[:]))
            population = sorted(population, key=lambda x: x[0])[:5]

    # --- Step 9: Simulated Annealing refinement ---
    md = best_md[:]
    rmd = best_rmd[:]
    current_cost = best_cost

    active_pqs = list(set(md[lq] for lq in logical_qubits if lq_combined[lq]))
    if len(active_pqs) < 2:
        active_pqs = physical_qubits

    n_sa_iters = max(4000, n_lq * 350)
    T_start = max(current_cost * 0.05, 0.5)
    T_end = max(current_cost * 0.00003, 1e-4)
    sa_alpha = (T_end / T_start) ** (1.0 / n_sa_iters)
    T = T_start
    no_improve = 0
    reheat_interval = n_sa_iters // 6
    n_reheats = 0

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
        if no_improve >= reheat_interval and n_reheats < 3:
            md = best_md[:]
            rmd = best_rmd[:]
            current_cost = best_cost
            T = T_start * (0.15 ** (n_reheats + 1))
            no_improve = 0
            n_reheats += 1

    # --- Step 10: Final full local search ---
    best_md, best_rmd = local_search_full(best_md, best_rmd, max_iters=600)

    self.mapping_dict = best_md
    self.reverse_mapping_dict = best_rmd

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)