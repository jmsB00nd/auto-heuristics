def init_mapping(self):
    """
    Spectral Flow Matching v3 — Procrustes-Aligned Multi-Seed QAP Solver.

    Key improvements over TIFN (696.08) and Spectral Flow v2:
    1. Orthogonal Procrustes alignment for spectral seed (optimal rotation, not just sign flip)
    2. Chain-aware interaction boosting: consecutive gate chains get transitive weight boost
    3. Cost-proportional SA sampling: bias moves toward high-cost qubit pairs
    4. Multi-sign spectral ensemble: try 2^k sign combinations for k eigenvectors
    5. Deeper critical path analysis with successor gate count weighting
    6. Chain-aware ILS perturbation strategy (hub relocation)
    7. Extended 2nd-order with decay, path-aware chain interactions
    8. More seeds, larger population, more ILS restarts and SA iterations
    """
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
        criticality = [3.5] * len(gates_list)
    else:
        criticality = [1.0 + 3.5 * (1.0 - s / max_slack) for s in gate_slack]

    # --- Step 2b: Successor count weighting ---
    qubit_last_2q_gate = {}
    gate_successors = [0] * len(gates_list)
    for idx in range(len(gates_list) - 1, -1, -1):
        _, qubits = gates_list[idx]
        if len(qubits) == 2:
            succ_count = 0
            for q in qubits:
                if q in qubit_last_2q_gate:
                    succ_count += gate_successors[qubit_last_2q_gate[q]] + 1
            gate_successors[idx] = succ_count
            for q in qubits:
                qubit_last_2q_gate[q] = idx

    max_succ = max(gate_successors) if gate_successors else 1
    if max_succ == 0:
        max_succ = 1

    # --- Step 3: First-order interaction weights ---
    interaction_weight = defaultdict(float)
    half_life = max(max_layer / 4.5, 3.5)
    critical_end = max(1, int(max_layer * 0.15))
    crit_half_life = max(critical_end / 3.0, 1.0)

    for idx, (_, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_earliest_start[idx]
            w = math.exp(-layer * math.log(2) / half_life) + 0.05
            if layer <= critical_end:
                w += 2.5 * math.exp(-layer * math.log(2) / crit_half_life)
            w *= criticality[idx]
            succ_boost = 1.0 + 0.3 * (gate_successors[idx] / max_succ)
            w *= succ_boost
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

    # --- Step 3b: Second-order + chain-aware boosting ---
    second_order = defaultdict(float)
    alpha2 = 0.18

    # Chain detection: if qubit q has gates q-A then q-B, boost A-B
    qubit_gate_seq = defaultdict(list)
    for idx, (_, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            qubit_gate_seq[q1].append((idx, q2))
            qubit_gate_seq[q2].append((idx, q1))

    chain_boost = defaultdict(float)
    for q in logical_qubits:
        seq = qubit_gate_seq[q]
        for i in range(len(seq) - 1):
            idx_i, partner_i = seq[i]
            for j in range(i + 1, min(i + 4, len(seq))):
                idx_j, partner_j = seq[j]
                if partner_i != partner_j:
                    key = (min(partner_i, partner_j), max(partner_i, partner_j))
                    layer_gap = abs(gate_earliest_start[idx_j] - gate_earliest_start[idx_i])
                    boost = 0.25 / (1.0 + layer_gap * 0.3)
                    iw_i = interaction_weight.get((min(q, partner_i), max(q, partner_i)), 0)
                    iw_j = interaction_weight.get((min(q, partner_j), max(q, partner_j)), 0)
                    chain_boost[key] += boost * math.sqrt(max(iw_i, 0.01) * max(iw_j, 0.01))

    for mid in logical_qubits:
        neighbors = list(lq_interactions[mid].items())
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                nb1, w1 = neighbors[i]
                nb2, w2 = neighbors[j]
                key = (min(nb1, nb2), max(nb1, nb2))
                second_order[key] += alpha2 * math.sqrt(w1 * w2)

    combined_weight = defaultdict(float)
    for k, v in interaction_weight.items():
        combined_weight[k] += v
    for k, v in second_order.items():
        combined_weight[k] += v
    for k, v in chain_boost.items():
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

    # --- Step 5: Rearrangement-inequality Hungarian cost matrix ---
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

    # --- Step 5b: Spectral embedding with Procrustes alignment + multi-sign ---
    def spectral_seed_procrustes():
        lq_index = {lq: i for i, lq in enumerate(logical_qubits)}
        pq_index = {pq: i for i, pq in enumerate(physical_qubits)}

        k_dim = min(6, n_lq - 1, n_pq - 1)
        if k_dim < 1:
            return []

        # Logical Laplacian (weighted)
        L_logical = np.zeros((n_lq, n_lq))
        for (q1, q2), w in interaction_weight.items():
            i, j = lq_index[q1], lq_index[q2]
            L_logical[i, j] -= w
            L_logical[j, i] -= w
            L_logical[i, i] += w
            L_logical[j, j] += w

        evals_l, evecs_l = np.linalg.eigh(L_logical)
        embed_l = evecs_l[:, 1:k_dim + 1].copy()

        # Hardware Laplacian
        L_hw = np.zeros((n_pq, n_pq))
        for pq in physical_qubits:
            for nb in self.backend[pq]:
                if nb in pq_index:
                    i, j = pq_index[pq], pq_index[nb]
                    if i < j:
                        L_hw[i, j] -= 1
                        L_hw[j, i] -= 1
                        L_hw[i, i] += 1
                        L_hw[j, j] += 1

        evals_h, evecs_h = np.linalg.eigh(L_hw)
        embed_h = evecs_h[:, 1:k_dim + 1].copy()

        # Scale by inverse sqrt of eigenvalue
        for d in range(k_dim):
            if evals_l[d + 1] > 1e-10:
                embed_l[:, d] /= math.sqrt(evals_l[d + 1])
            if evals_h[d + 1] > 1e-10:
                embed_h[:, d] /= math.sqrt(evals_h[d + 1])

        results = []

        # Method 1: Procrustes alignment
        raw_cost = np.zeros((n_lq, n_pq))
        for i in range(n_lq):
            for j in range(n_pq):
                raw_cost[i, j] = np.sum((embed_l[i] - embed_h[j]) ** 2)
        r0, c0 = linear_sum_assignment(raw_cost)
        anchor_map = {r: c for r, c in zip(r0, c0)}

        n_anchors = min(n_lq, n_pq)
        A = np.zeros((n_anchors, k_dim))
        B = np.zeros((n_anchors, k_dim))
        for idx_a, (r, c) in enumerate(anchor_map.items()):
            if idx_a >= n_anchors:
                break
            A[idx_a] = embed_l[r]
            B[idx_a] = embed_h[c]

        try:
            U, _, Vt = np.linalg.svd(A.T @ B)
            R = U @ Vt
            embed_l_rotated = embed_l @ R

            proc_cost = np.zeros((n_lq, n_pq))
            for i in range(n_lq):
                for j in range(n_pq):
                    proc_cost[i, j] = np.sum((embed_l_rotated[i] - embed_h[j]) ** 2)

            r_ind, c_ind = linear_sum_assignment(proc_cost)
            results.append({logical_qubits[r]: physical_qubits[c] for r, c in zip(r_ind, c_ind)})
        except Exception:
            pass

        # Method 2: Multi-sign enumeration (first 3 dimensions)
        n_sign_dims = min(3, k_dim)
        for sign_mask in range(1, 2 ** n_sign_dims):
            embed_l_signed = embed_l.copy()
            for d in range(n_sign_dims):
                if sign_mask & (1 << d):
                    embed_l_signed[:, d] = -embed_l_signed[:, d]

            sign_cost = np.zeros((n_lq, n_pq))
            for i in range(n_lq):
                for j in range(n_pq):
                    sign_cost[i, j] = np.sum((embed_l_signed[i] - embed_h[j]) ** 2)

            r_ind, c_ind = linear_sum_assignment(sign_cost)
            results.append({logical_qubits[r]: physical_qubits[c] for r, c in zip(r_ind, c_ind)})

        # Method 3: Original (no flip)
        r_ind, c_ind = linear_sum_assignment(raw_cost)
        results.append({logical_qubits[r]: physical_qubits[c] for r, c in zip(r_ind, c_ind)})

        return results

    try:
        spectral_maps = spectral_seed_procrustes()
    except Exception:
        spectral_maps = []

    # --- Step 6: Greedy BFS seed ---
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

    # --- Step 8: Cost utilities ---
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

    # --- Step 9: Two-tier local search ---
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

    # --- Step 10: Multi-start seeds ---
    sorted_combined = sorted(combined_weight.items(), key=lambda x: x[1], reverse=True)
    alt_pairs = [pair for pair, _ in sorted_combined[1:5]] if len(sorted_combined) > 1 else []

    seed_configs = [
        ("hungarian", hungarian_map, None),
    ]
    for smap in spectral_maps:
        seed_configs.append(("spectral", smap, None))

    seed_configs.append(("bfs_default", None, (None, None)))
    for anchor in top_central[:4]:
        seed_configs.append(("bfs_central", None, (anchor, None)))
    for anchor in top_degree_pq[:2]:
        seed_configs.append(("bfs_degree", None, (anchor, None)))
    for pair in alt_pairs:
        seed_configs.append(("bfs_pair", None, (None, pair)))

    best_md, best_rmd, best_cost = None, None, float('inf')
    population = []

    for name, fixed_map, bfs_args in seed_configs:
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

    population = sorted(population, key=lambda x: x[0])[:7]

    # --- Step 11: ILS with seven perturbation strategies ---
    rng = random.Random(42)
    n_restarts = min(50, max(12, n_lq // 2))

    for restart in range(n_restarts):
        base_idx = restart % len(population)
        _, base_md, base_rmd = population[base_idx]
        md = base_md[:]
        rmd = base_rmd[:]
        strategy = restart % 7

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
            # Targeted worst-pair relocation
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
                for pp1 in top_central[:max(8, n_pq // 4)]:
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
            # Subgraph re-embedding
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

        elif strategy == 4:
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

        elif strategy == 5:
            # 3-opt cyclic perturbation
            n_triples = rng.randint(1, max(2, n_lq // 6))
            candidates = lq_by_degree[:max(6, n_lq // 2)]
            for _ in range(n_triples):
                if len(candidates) < 3:
                    break
                triple = rng.sample(candidates, 3)
                p0, p1_t, p2_t = md[triple[0]], md[triple[1]], md[triple[2]]
                do_swap(p0, p1_t, md, rmd)
                do_swap(p1_t, p2_t, md, rmd)

        else:
            # Strategy 6: Chain-aware hub relocation
            if interaction_weight:
                worst_chains = []
                for q in logical_qubits:
                    chain_cost = 0.0
                    for nb, w in lq_combined[q].items():
                        chain_cost += w * self.distance_matrix[md[q]][md[nb]]
                    worst_chains.append((chain_cost, q))
                worst_chains.sort(reverse=True)

                hub_lq = worst_chains[0][1]
                hub_nbs = sorted(lq_combined[hub_lq].keys(),
                                key=lambda nb: lq_combined[hub_lq][nb], reverse=True)
                cluster = [hub_lq] + hub_nbs[:min(3, len(hub_nbs))]

                best_center_cost = float('inf')
                best_center = md[hub_lq]
                for pq in top_central[:max(6, n_pq // 5)]:
                    cc = sum(
                        lq_combined[hub_lq].get(nb, 0) * self.distance_matrix[pq][md[nb]]
                        for nb in lq_combined[hub_lq]
                        if nb not in cluster
                    )
                    if cc < best_center_cost:
                        best_center_cost = cc
                        best_center = pq

                cur = md[hub_lq]
                if cur != best_center:
                    do_swap(cur, best_center, md, rmd)
                for nb_lq in hub_nbs[:min(3, len(hub_nbs))]:
                    best_nb_pq = md[nb_lq]
                    best_nb_cost = float('inf')
                    for adj_pq in pq_adj.get(md[hub_lq], []):
                        c = sum(
                            lq_combined[nb_lq].get(other, 0) * self.distance_matrix[adj_pq][md[other]]
                            for other in lq_combined[nb_lq]
                        )
                        if c < best_nb_cost:
                            best_nb_cost = c
                            best_nb_pq = adj_pq
                    cur_nb = md[nb_lq]
                    if cur_nb != best_nb_pq:
                        do_swap(cur_nb, best_nb_pq, md, rmd)

        md, rmd = local_search(md, rmd)
        c = qap_cost(md)
        if c < best_cost:
            best_cost = c
            best_md = md[:]
            best_rmd = rmd[:]
        if len(population) < 7 or c < population[-1][0]:
            population.append((c, md[:], rmd[:]))
            population = sorted(population, key=lambda x: x[0])[:7]

    # --- Step 12: Extended SA with cost-proportional sampling + warm restarts ---
    md = best_md[:]
    rmd = best_rmd[:]
    current_cost = best_cost

    active_pqs = list(set(md[lq] for lq in logical_qubits if lq_combined[lq]))
    if len(active_pqs) < 2:
        active_pqs = physical_qubits

    # Build per-physical-qubit cost contribution for biased sampling
    def build_pq_weights(cur_md, cur_rmd):
        pq_cost_map = {}
        for pq in active_pqs:
            lq = cur_rmd[pq]
            c = 0.0
            for nb, w in lq_combined.get(lq, {}).items():
                c += w * self.distance_matrix[pq][cur_md[nb]]
            pq_cost_map[pq] = c + 0.01
        total_c = sum(pq_cost_map.values())
        weights = [pq_cost_map[pq] / total_c for pq in active_pqs]
        cum = []
        s = 0.0
        for w in weights:
            s += w
            cum.append(s)
        return cum

    cum_weights = build_pq_weights(md, rmd)

    def weighted_sample_pq():
        r = rng.random() * cum_weights[-1]
        lo, hi = 0, len(cum_weights) - 1
        while lo < hi:
            mid_idx = (lo + hi) // 2
            if cum_weights[mid_idx] < r:
                lo = mid_idx + 1
            else:
                hi = mid_idx
        return active_pqs[lo]

    n_sa_iters = max(6000, n_lq * 500)
    T_start = max(current_cost * 0.07, 0.5)
    T_end = max(current_cost * 0.00002, 1e-4)
    sa_alpha = (T_end / T_start) ** (1.0 / n_sa_iters)
    T = T_start

    no_improve_streak = 0
    reheat_interval = n_sa_iters // 8
    n_reheats = 0
    max_reheats = 5
    weight_refresh = n_sa_iters // 10

    for sa_iter in range(n_sa_iters):
        # Alternate between cost-proportional and uniform sampling
        if sa_iter % 3 == 0:
            p1 = weighted_sample_pq()
            p2 = rng.choice(active_pqs)
            while p2 == p1:
                p2 = rng.choice(active_pqs)
        else:
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

        T *= sa_alpha

        # Periodically refresh sampling weights
        if sa_iter > 0 and sa_iter % weight_refresh == 0:
            cum_weights = build_pq_weights(md, rmd)

        # Warm restart: reset to best solution and reheat
        if no_improve_streak >= reheat_interval and n_reheats < max_reheats:
            md = best_md[:]
            rmd = best_rmd[:]
            current_cost = best_cost
            T = T_start * (0.12 ** (n_reheats + 1))
            no_improve_streak = 0
            n_reheats += 1
            cum_weights = build_pq_weights(md, rmd)

    # --- Step 13: Final full local search ---
    best_md, best_rmd = local_search_full(best_md, best_rmd, max_iters=700)

    self.mapping_dict = best_md
    self.reverse_mapping_dict = best_rmd

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)