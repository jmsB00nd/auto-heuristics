def init_mapping(self):
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import math
    import random

    # --- Step 1: True circuit depth via topological scheduling ---
    gates_list = list(self.access.items())

    logical_qubit_set = set()
    for _, qubits in gates_list:
        for q in qubits:
            logical_qubit_set.add(q)

    qubit_ready = {}
    gate_layers = []
    for _, qubits in gates_list:
        layer = max((qubit_ready.get(q, 0) for q in qubits), default=0)
        gate_layers.append(layer)
        for q in qubits:
            qubit_ready[q] = layer + 1

    max_layer = max(gate_layers, default=0)

    # --- Step 2: Layer-decayed + critical-window interaction weights ---
    interaction_weight = defaultdict(float)
    half_life = max(max_layer / 3.5, 5.0)
    critical_end = max(1, int(max_layer * 0.15))
    crit_half_life = max(critical_end / 2.0, 1.0)

    for idx, (_, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_layers[idx]
            w = math.exp(-layer * math.log(2) / half_life) + 0.08
            if layer <= critical_end:
                w += 1.5 * math.exp(-layer * math.log(2) / crit_half_life)
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

    pq_adj = {pq: [nb for nb in self.backend.get(pq, []) if nb in pq_set]
              for pq in physical_qubits}
    pq_degree = {pq: len(pq_adj[pq]) for pq in physical_qubits}

    # --- Step 3: Physical topology analysis ---
    # Closeness centrality: sum of inverse distances to all other qubits
    pq_centrality = {}
    for pq in physical_qubits:
        c = sum(
            1.0 / self.distance_matrix[pq][other]
            for other in physical_qubits
            if other != pq and self.distance_matrix[pq][other] not in (0, float('inf'))
        )
        pq_centrality[pq] = c

    # Precompute radius-2 physical neighborhoods (used in fast local search)
    pq_r2 = {}
    for pq in physical_qubits:
        nbrs = set(pq_adj[pq])
        for nb in pq_adj[pq]:
            nbrs.update(pq_adj[nb])
        nbrs.discard(pq)
        pq_r2[pq] = list(nbrs)

    # --- Step 4: Rearrangement-inequality cost matrix → Hungarian ---
    lq_sorted_weights = {
        lq: sorted(lq_interactions[lq].values(), reverse=True)
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

    # --- Step 5: Greedy BFS seed with optional physical anchor ---
    def greedy_bfs_seed(anchor_pq=None, seed_pair=None):
        lq_phys = {}
        phys_used = set()

        # Choose the logical seed pair
        if seed_pair is not None:
            slq1, slq2 = seed_pair
        elif interaction_weight:
            slq1, slq2 = max(interaction_weight, key=interaction_weight.__getitem__)
        else:
            slq1 = logical_qubits[0]
            slq2 = logical_qubits[1] if n_lq > 1 else logical_qubits[0]

        # Choose the physical seed pair
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

        # Orient: higher-interaction logical → higher-degree physical
        w1 = sum(lq_interactions[slq1].values())
        w2 = sum(lq_interactions[slq2].values())
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
                s = sum(lq_interactions[lq].get(p, 0) for p in placed)
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
                    for nb, w in lq_interactions[best_lq].items()
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

    # --- Step 6: Build bijection ---
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

    # --- Step 7: Cost utilities ---
    def qap_cost(md):
        return sum(
            w * self.distance_matrix[md[q1]][md[q2]]
            for (q1, q2), w in interaction_weight.items()
        )

    def swap_delta(p1, p2, md, rmd):
        a, b = rmd[p1], rmd[p2]
        delta = 0.0
        for nb, w in lq_interactions[a].items():
            pp = md[nb]
            if pp == p2:
                continue
            delta += w * (self.distance_matrix[p2][pp] - self.distance_matrix[p1][pp])
        for nb, w in lq_interactions[b].items():
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
        key=lambda lq: sum(lq_interactions[lq].values()),
        reverse=True
    )

    # --- Step 8: Two-tier local search ---
    # Fast tier: check radius-2 physical neighbors of current position.
    # Slow tier (every `full_every` iterations): check all physical qubits.
    def local_search(md, rmd, max_iters=400, full_every=4):
        for iteration in range(max_iters):
            improved = False
            do_full = (iteration % full_every == 0)
            for lq in lq_by_degree:
                p1 = md[lq]
                best_d, best_p2 = -1e-9, -1

                # Always check radius-2 neighborhood
                for p2 in pq_r2[p1]:
                    d = swap_delta(p1, p2, md, rmd)
                    if d < best_d:
                        best_d, best_p2 = d, p2

                # Periodically check globally
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

    # --- Step 9: Multi-start seeds ---
    # Hungarian + BFS from default anchor + top-centrality anchors + top-degree anchors
    # Also try the 2nd-highest interaction pair as alternative BFS seed pair
    top_central = sorted(physical_qubits, key=lambda p: pq_centrality[p], reverse=True)
    top_degree = sorted(physical_qubits, key=lambda p: pq_degree[p], reverse=True)

    # Top-2 interaction pairs for seeding
    sorted_interactions = sorted(interaction_weight.items(), key=lambda x: x[1], reverse=True)
    alt_pairs = [pair for pair, _ in sorted_interactions[1:3]] if len(sorted_interactions) > 1 else []

    seed_configs = [
        (hungarian_map, None),  # Hungarian solution
        (None, (None, None)),   # Default BFS
    ]
    for anchor in top_central[:2]:
        seed_configs.append((None, (anchor, None)))
    for anchor in top_degree[:2]:
        seed_configs.append((None, (anchor, None)))
    for pair in alt_pairs:
        seed_configs.append((None, (None, pair)))

    best_md, best_rmd, best_cost = None, None, float('inf')
    for fixed_map, bfs_args in seed_configs:
        if fixed_map is not None:
            md, rmd = make_mapping(fixed_map)
        else:
            anchor_pq, seed_pair = bfs_args
            bfs_map = greedy_bfs_seed(anchor_pq=anchor_pq, seed_pair=seed_pair)
            md, rmd = make_mapping(bfs_map)
        md, rmd = local_search_full(md, rmd)
        c = qap_cost(md)
        if c < best_cost:
            best_cost = c
            best_md, best_rmd = md[:], rmd[:]

    # --- Step 10: ILS with four diversified perturbation strategies ---
    rng = random.Random(42)
    n_restarts = min(25, max(6, n_lq // 2))

    for restart in range(n_restarts):
        md = best_md[:]
        rmd = best_rmd[:]
        strategy = restart % 4

        if strategy == 0:
            # Cyclic rotation of a random subset
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
            # Random scatter: swap random pairs
            n_swaps = rng.randint(2, max(3, n_lq // 3))
            pool = rng.sample(logical_qubits, min(n_swaps * 2, n_lq))
            for i in range(0, len(pool) - 1, 2):
                do_swap(md[pool[i]], md[pool[i + 1]], md, rmd)

        elif strategy == 2:
            # Targeted perturbation: perturb worst-cost interactions
            worst = sorted(
                interaction_weight.items(),
                key=lambda x: x[1] * self.distance_matrix[md[x[0][0]]][md[x[0][1]]],
                reverse=True
            )[:max(2, n_lq // 4)]
            perturbed = set()
            for (q1, q2), _ in worst:
                if q1 not in perturbed and q2 not in perturbed:
                    candidates = [lq for lq in logical_qubits
                                  if lq not in perturbed and lq != q1 and lq != q2]
                    if candidates:
                        q3 = rng.choice(candidates)
                        do_swap(md[q2], md[q3], md, rmd)
                        perturbed.update([q1, q2, q3])

        else:
            # Subgraph relocation: pick a logically-connected cluster (BFS in interaction graph)
            # and physically relocate it to a BFS-expanded cluster around a random central node.
            seed_lq = rng.choice(lq_by_degree[:max(1, n_lq // 4)])
            sub_size = rng.randint(2, max(3, n_lq // 5))

            # BFS expansion in logical interaction graph
            subgraph = [seed_lq]
            lq_frontier = sorted(
                lq_interactions[seed_lq].keys(),
                key=lambda q: lq_interactions[seed_lq][q],
                reverse=True
            )
            for lq in lq_frontier:
                if len(subgraph) >= sub_size:
                    break
                if lq not in subgraph:
                    subgraph.append(lq)

            # Pick a random high-centrality physical anchor
            cand_anchors = top_central[:max(3, n_pq // 6)]
            new_anchor = rng.choice(cand_anchors)

            # BFS-expand physical cluster from anchor, avoiding qubits used by non-subgraph lqs
            phys_blocked = set(md[lq] for lq in logical_qubits if lq not in subgraph)
            new_positions = []
            phys_frontier = [new_anchor]
            visited_pq = {new_anchor}
            for pq in phys_frontier:
                if pq not in phys_blocked:
                    new_positions.append(pq)
                if len(new_positions) >= len(subgraph):
                    break
                for nb in pq_adj[pq]:
                    if nb not in visited_pq:
                        visited_pq.add(nb)
                        phys_frontier.append(nb)

            # Fallback: fill from unblocked physical qubits
            if len(new_positions) < len(subgraph):
                for pq in physical_qubits:
                    if pq not in phys_blocked and pq not in set(new_positions):
                        new_positions.append(pq)
                        if len(new_positions) >= len(subgraph):
                            break

            # Relocate via swaps
            for i, lq in enumerate(subgraph):
                if i < len(new_positions):
                    tgt = new_positions[i]
                    cur = md[lq]
                    if cur != tgt:
                        do_swap(cur, tgt, md, rmd)

        md, rmd = local_search(md, rmd)
        c = qap_cost(md)
        if c < best_cost:
            best_cost = c
            best_md = md[:]
            best_rmd = rmd[:]

    # --- Step 11: Final full local search on best solution ---
    best_md, best_rmd = local_search_full(best_md, best_rmd, max_iters=500)

    self.mapping_dict = best_md
    self.reverse_mapping_dict = best_rmd

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)