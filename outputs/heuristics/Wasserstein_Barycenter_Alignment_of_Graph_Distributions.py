def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import random
    import math

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    dist = self.distance_matrix

    # Extract logical qubits used in circuit
    logical_qubit_set = set()
    for qubits in self.access.values():
        logical_qubit_set.update(qubits)
    logical_qubits = sorted(logical_qubit_set)
    num_logical = len(logical_qubits)

    # Trivial fallback
    if num_logical == 0:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        return

    # Build interaction weights between logical qubits
    interaction = {}
    for g, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            key = (min(q1, q2), max(q1, q2))
            interaction[key] = interaction.get(key, 0) + 1

    # Adjacency list for logical interaction graph
    logical_neighbors = {lq: {} for lq in logical_qubits}
    for (q1, q2), w in interaction.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    num_phys = len(physical_qubits)
    pq_index = {pq: i for i, pq in enumerate(physical_qubits)}

    # Precompute distance submatrix for physical qubits
    dist_sub = np.zeros((num_phys, num_phys))
    for i, pi in enumerate(physical_qubits):
        for j, pj in enumerate(physical_qubits):
            dist_sub[i][j] = dist[pi][pj]

    # Step 1: Self-consistent demand distribution iteration
    # d_l(p) for each logical qubit l over physical qubits
    # Initialize uniform
    demands = {}
    for lq in logical_qubits:
        demands[lq] = np.ones(num_phys) / num_phys

    max_dist = dist_sub.max() if dist_sub.size > 0 else 1.0
    sigma = max(max_dist / 3.0, 1.0)

    for iteration in range(7):
        new_demands = {}
        for lq in logical_qubits:
            neighbors = logical_neighbors.get(lq, {})
            if not neighbors:
                new_demands[lq] = np.ones(num_phys) / num_phys
                continue

            d_l = np.zeros(num_phys)
            for lq_prime, w in neighbors.items():
                # Compute centroid of lq_prime's current distribution
                centroid_dist = demands[lq_prime] @ dist_sub  # shape (num_phys,)
                # d_l(p) += w * exp(-centroid_dist(p) / sigma)
                d_l += w * np.exp(-centroid_dist / sigma)

            total = d_l.sum()
            if total > 0:
                d_l /= total
            else:
                d_l = np.ones(num_phys) / num_phys
            new_demands[lq] = d_l
        demands = new_demands

    # Step 2-3: Build cost matrix C(l,p) = W_1(d_l, delta_p) = sum_p' d_l(p') * dist(p, p')
    cost_matrix = np.zeros((num_logical, num_phys))
    for i, lq in enumerate(logical_qubits):
        # C(l, p) = sum_p' d_l(p') * dist(p, p') = dist_sub[j, :] @ d_l for each j
        cost_matrix[i] = dist_sub @ demands[lq]

    # Pad if num_logical < num_phys (rectangular assignment)
    # scipy handles rectangular matrices directly
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build initial mapping from Hungarian result
    mapping = [-1] * num_q
    reverse_mapping = [-1] * num_q
    used_physical = set()

    for r, c in zip(row_ind, col_ind):
        lq = logical_qubits[r]
        pq = physical_qubits[c]
        mapping[lq] = pq
        reverse_mapping[pq] = lq
        used_physical.add(pq)

    # Assign unmapped logical qubits (if any) to remaining physical qubits
    remaining_physical = [pq for pq in physical_qubits if pq not in used_physical]
    unmapped_logical = [lq for lq in range(num_q) if mapping[lq] == -1]
    for lq, pq in zip(unmapped_logical, remaining_physical):
        mapping[lq] = pq
        reverse_mapping[pq] = lq

    # Step 4: ILS + SA refinement with routing-simulation cost
    def compute_cost(m):
        total = 0.0
        for (q1, q2), w in interaction.items():
            total += w * dist[m[q1]][m[q2]]
        return total

    def do_swap(m, rm, pq_a, pq_b):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        m[lq_a], m[lq_b] = pq_b, pq_a
        rm[pq_a], rm[pq_b] = lq_b, lq_a

    best_m = mapping[:]
    best_rm = reverse_mapping[:]
    best_cost = compute_cost(best_m)

    current_m = mapping[:]
    current_rm = reverse_mapping[:]
    current_cost = best_cost

    # Collect swap candidates (edges in hardware graph)
    swap_edges = []
    for pq in physical_qubits:
        for neighbor in self.backend.get(pq, []):
            if pq < neighbor:
                swap_edges.append((pq, neighbor))

    if not swap_edges:
        self.mapping_dict = best_m
        self.reverse_mapping_dict = best_rm
        return

    # SA + ILS parameters
    num_restarts = 3
    sa_steps = min(2000, max(500, num_logical * 30))

    for restart in range(num_restarts):
        if restart > 0:
            # Perturbation: random swaps from best solution
            current_m = best_m[:]
            current_rm = best_rm[:]
            num_perturb = max(2, num_logical // 5)
            for _ in range(num_perturb):
                pq_a, pq_b = random.choice(swap_edges)
                do_swap(current_m, current_rm, pq_a, pq_b)
            current_cost = compute_cost(current_m)

        T = current_cost * 0.15 + 1.0
        T_min = 0.01
        alpha = (T_min / T) ** (1.0 / sa_steps) if T > T_min else 0.99

        for step in range(sa_steps):
            pq_a, pq_b = random.choice(swap_edges)
            lq_a, lq_b = current_rm[pq_a], current_rm[pq_b]

            # Incremental cost delta
            delta = 0.0
            for lq_x, pq_new in [(lq_a, pq_b), (lq_b, pq_a)]:
                if lq_x < 0:
                    continue
                for lq_n, w in logical_neighbors.get(lq_x, {}).items():
                    pq_n = current_m[lq_n]
                    if lq_n == lq_b or lq_n == lq_a:
                        continue
                    pq_old = pq_a if lq_x == lq_a else pq_b
                    delta += w * (dist[pq_new][pq_n] - dist[pq_old][pq_n])
            # Cross-term between lq_a and lq_b
            if lq_a >= 0 and lq_b >= 0:
                w_ab = logical_neighbors.get(lq_a, {}).get(lq_b, 0)
                if w_ab > 0:
                    old_d = dist[pq_a][pq_b]
                    new_d = dist[pq_b][pq_a]  # same due to symmetry
                    delta += w_ab * (new_d - old_d)  # = 0, symmetric

            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                do_swap(current_m, current_rm, pq_a, pq_b)
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_m = current_m[:]
                    best_rm = current_rm[:]

            T *= alpha

    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm