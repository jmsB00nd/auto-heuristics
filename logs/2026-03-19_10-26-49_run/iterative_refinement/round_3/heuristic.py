def init_mapping(self):
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import math
    import random

    # --- Step 1: Layer-decayed interaction weights ---
    # Interactions early in the circuit get exponentially higher weight:
    # the initial mapping most directly impacts the first execution layers.
    interaction_weight = defaultdict(float)
    logical_qubit_set = set()

    gates_list = list(self.access.items())
    n_gates = len(gates_list)
    half_life = max(n_gates / 4.0, 8.0)  # weight halves every n_gates/4 gates

    for idx, (gate, qubits) in enumerate(gates_list):
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            w = math.exp(-idx * math.log(2) / half_life) + 0.1  # +0.1 baseline keeps late gates relevant
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

    # --- Step 2: Build logical adjacency dict for local search ---
    lq_interactions = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        lq_interactions[q1][q2] = w
        lq_interactions[q2][q1] = w

    # --- Step 3: Rearrangement-inequality cost matrix ---
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

    cost_matrix = np.zeros((n_lq, len(physical_qubits)))
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

    # --- Step 4: Hungarian assignment ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    lq_to_phys = {
        logical_qubits[r]: physical_qubits[c]
        for r, c in zip(row_ind, col_ind)
    }

    # --- Step 5: Utility functions ---
    def make_bijection(lq_phys_map):
        md = list(range(self.num_qubits))
        rmd = list(range(self.num_qubits))
        for lq, tgt in lq_phys_map.items():
            cur = md[lq]
            if cur == tgt:
                continue
            disp = rmd[tgt]
            md[lq] = tgt
            md[disp] = cur
            rmd[tgt] = lq
            rmd[cur] = disp
        return md, rmd

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

    # --- Step 6: 2-opt local search with degree-sorted scan order ---
    # Scanning the most-connected qubits first concentrates improvement
    # where it matters most and accelerates convergence per iteration.
    lq_by_degree = sorted(
        logical_qubits,
        key=lambda lq: sum(lq_interactions[lq].values()),
        reverse=True
    )

    def local_search(md, rmd, max_iters=250):
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

    # --- Step 7: Iterated Local Search (ILS) ---
    # Perturbation: cyclic rotation of a random subset of logical qubit positions.
    # Rotation guarantees a structural change unreachable by 2-opt alone,
    # enabling escape from local optima (similar to double-bridge in TSP ILS).
    mapping_dict, reverse_mapping_dict = make_bijection(lq_to_phys)
    mapping_dict, reverse_mapping_dict = local_search(mapping_dict, reverse_mapping_dict)

    best_md = mapping_dict[:]
    best_rmd = reverse_mapping_dict[:]
    best_cost = qap_cost(mapping_dict)

    rng = random.Random(42)
    n_restarts = min(10, max(3, n_lq // 2))

    for _ in range(n_restarts):
        md = best_md[:]
        rmd = best_rmd[:]

        # Cyclic rotation of k qubits: unreachable by any 2-opt move
        k = rng.randint(max(3, n_lq // 4), max(4, n_lq // 2 + 1))
        k = min(k, n_lq)
        sample = rng.sample(logical_qubits, k)
        positions = [md[lq] for lq in sample]
        offset = rng.randint(1, k - 1)
        rotated = positions[offset:] + positions[:offset]

        # Apply permutation via sequential swaps (each do_swap maintains bijection)
        for lq, tgt in zip(sample, rotated):
            cur = md[lq]
            if cur != tgt:
                do_swap(cur, tgt, md, rmd)

        md, rmd = local_search(md, rmd)
        c = qap_cost(md)
        if c < best_cost:
            best_cost = c
            best_md = md[:]
            best_rmd = rmd[:]

    self.mapping_dict = best_md
    self.reverse_mapping_dict = best_rmd

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)