def init_mapping(self):
    import numpy as np
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment
    import random as rng

    rng.seed(42)
    np.random.seed(42)

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # --- Step 1: Build interaction graph with weights ---
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)

    # Topological ordering for temporal decay
    total_gates = max(len(self.access), 1)
    gate_list = sorted(self.access.keys())
    gate_rank = {g: i for i, g in enumerate(gate_list)}
    alpha = 2.0

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            w = np.exp(-alpha * gate_rank.get(gate, 0) / total_gates)
            interaction_weight[key] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)
    n_logical = len(logical_qubits)
    n_physical = len(physical_qubits)

    if n_logical == 0:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            from src.utils.python_to_isl import dict_to_isl_map
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 2: MDS embedding of physical qubits into 2D ---
    dist = np.zeros((n_physical, n_physical))
    pq_idx = {pq: i for i, pq in enumerate(physical_qubits)}
    for i, pq1 in enumerate(physical_qubits):
        for j, pq2 in enumerate(physical_qubits):
            dist[i][j] = self.distance_matrix[pq1][pq2]

    # Classical MDS
    n = n_physical
    H = np.eye(n) - np.ones((n, n)) / n
    D_sq = dist ** 2
    B = -0.5 * H @ D_sq @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top 2 dimensions
    dim = 2
    pos_eigenvalues = np.maximum(eigenvalues[:dim], 0)
    phys_coords = eigenvectors[:, :dim] * np.sqrt(pos_eigenvalues)

    # --- Step 3: Initialize logical qubit positions ---
    # Place logical qubits near physical positions based on degree centrality
    lq_idx = {lq: i for i, lq in enumerate(logical_qubits)}

    # Initialize logical positions: spread around center of physical coords
    center = phys_coords.mean(axis=0)
    logical_coords = np.zeros((n_logical, dim))

    # Use interaction-weighted initialization: place near high-connectivity physical region
    for i, lq in enumerate(logical_qubits):
        # Start near a random physical qubit position with small noise
        rand_pq = rng.randint(0, n_physical - 1)
        logical_coords[i] = phys_coords[rand_pq] + np.random.randn(dim) * 0.1

    # --- Step 4: Force-directed simulation ---
    T = 200
    initial_step = np.max(dist) * 0.1
    min_step = 1e-6

    for iteration in range(T):
        t = 1.0 - iteration / T  # cooling factor
        step_size = initial_step * t + min_step

        forces = np.zeros((n_logical, dim))

        # (1) Attractive forces between interacting logical qubits
        for (q1, q2), w in interaction_weight.items():
            if q1 not in lq_idx or q2 not in lq_idx:
                continue
            i1, i2 = lq_idx[q1], lq_idx[q2]
            delta = logical_coords[i2] - logical_coords[i1]
            d = np.linalg.norm(delta)
            if d < 1e-8:
                delta = np.random.randn(dim) * 0.01
                d = np.linalg.norm(delta)
            direction = delta / d
            # Attractive: pull toward adjacency distance (~1 in graph)
            attractive_mag = w * (d - 0.5)
            forces[i1] += attractive_mag * direction
            forces[i2] -= attractive_mag * direction

        # (2) Repulsive forces between all logical qubit pairs
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                delta = logical_coords[j] - logical_coords[i]
                d = np.linalg.norm(delta)
                if d < 1e-8:
                    delta = np.random.randn(dim) * 0.01
                    d = np.linalg.norm(delta)
                direction = delta / d
                repulsive_mag = 1.0 / (d * d)
                forces[i] -= repulsive_mag * direction
                forces[j] += repulsive_mag * direction

        # (3) Anchoring force toward nearest physical qubit position
        anchor_strength = 0.05 + 0.95 * (1.0 - t)  # increases over iterations
        for i in range(n_logical):
            dists_to_phys = np.linalg.norm(phys_coords - logical_coords[i], axis=1)
            nearest_pq_idx = np.argmin(dists_to_phys)
            delta = phys_coords[nearest_pq_idx] - logical_coords[i]
            forces[i] += anchor_strength * delta

        # Apply forces with step size
        for i in range(n_logical):
            f_mag = np.linalg.norm(forces[i])
            if f_mag > 0:
                logical_coords[i] += step_size * forces[i] / max(f_mag, step_size)

    # --- Step 5: Hungarian assignment (snap to physical qubits) ---
    cost_matrix = np.zeros((n_logical, n_physical))
    for i in range(n_logical):
        for j in range(n_physical):
            cost_matrix[i][j] = np.linalg.norm(logical_coords[i] - phys_coords[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q

    used_physical = set()
    for r, c in zip(row_ind, col_ind):
        lq = logical_qubits[r]
        pq = physical_qubits[c]
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq
        used_physical.add(pq)

    # Fill unmapped logical qubits
    free_physical = [pq for pq in physical_qubits if pq not in used_physical]
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    # --- Step 6: ILS + SA refinement with interaction-weighted cost ---
    def compute_cost(m):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    best_mapping = mapping_dict[:]
    best_reverse = reverse_mapping_dict[:]
    best_cost = compute_cost(best_mapping)

    current_mapping = best_mapping[:]
    current_reverse = best_reverse[:]
    current_cost = best_cost

    # SA parameters
    sa_temp = max(best_cost * 0.3, 1.0)
    sa_cooling = 0.97
    sa_iterations = min(n_logical * n_logical * 3, 2000)

    placed_logical = logical_qubits[:]

    for _ in range(sa_iterations):
        if len(placed_logical) < 2:
            break

        # Pick two random logical qubits to swap
        idx_a = rng.randint(0, len(placed_logical) - 1)
        idx_b = rng.randint(0, len(placed_logical) - 2)
        if idx_b >= idx_a:
            idx_b += 1
        lq_a = placed_logical[idx_a]
        lq_b = placed_logical[idx_b]

        # Compute delta cost
        pq_a = current_mapping[lq_a]
        pq_b = current_mapping[lq_b]

        old_cost_a = 0.0
        old_cost_b = 0.0
        new_cost_a = 0.0
        new_cost_b = 0.0

        for (q1, q2), w in interaction_weight.items():
            if q1 == lq_a or q2 == lq_a:
                other = q2 if q1 == lq_a else q1
                if current_mapping[other] >= 0:
                    old_cost_a += w * self.distance_matrix[pq_a][current_mapping[other]]
                    if other == lq_b:
                        new_cost_a += w * self.distance_matrix[pq_b][pq_a]
                    else:
                        new_cost_a += w * self.distance_matrix[pq_b][current_mapping[other]]
            if q1 == lq_b or q2 == lq_b:
                other = q2 if q1 == lq_b else q1
                if other == lq_a:
                    continue  # already counted
                if current_mapping[other] >= 0:
                    old_cost_b += w * self.distance_matrix[pq_b][current_mapping[other]]
                    new_cost_b += w * self.distance_matrix[pq_a][current_mapping[other]]

        delta = (new_cost_a + new_cost_b) - (old_cost_a + old_cost_b)

        if delta < 0 or rng.random() < np.exp(-delta / max(sa_temp, 1e-10)):
            # Accept swap
            current_mapping[lq_a] = pq_b
            current_mapping[lq_b] = pq_a
            current_reverse[pq_a] = lq_b
            current_reverse[pq_b] = lq_a
            current_cost += delta

            if current_cost < best_cost:
                best_cost = current_cost
                best_mapping = current_mapping[:]
                best_reverse = current_reverse[:]

        sa_temp *= sa_cooling

    # ILS perturbation restarts
    for _ in range(5):
        perturbed_mapping = best_mapping[:]
        perturbed_reverse = best_reverse[:]

        # Random perturbation: swap k random pairs
        k = max(2, n_logical // 10)
        for _ in range(k):
            if len(placed_logical) < 2:
                break
            idx_a = rng.randint(0, len(placed_logical) - 1)
            idx_b = rng.randint(0, len(placed_logical) - 2)
            if idx_b >= idx_a:
                idx_b += 1
            lq_a = placed_logical[idx_a]
            lq_b = placed_logical[idx_b]
            pq_a = perturbed_mapping[lq_a]
            pq_b = perturbed_mapping[lq_b]
            perturbed_mapping[lq_a] = pq_b
            perturbed_mapping[lq_b] = pq_a
            perturbed_reverse[pq_a] = lq_b
            perturbed_reverse[pq_b] = lq_a

        # Local search from perturbed solution
        p_cost = compute_cost(perturbed_mapping)
        improved = True
        rounds = 0
        while improved and rounds < 3:
            improved = False
            rounds += 1
            for i in range(len(placed_logical)):
                for j in range(i + 1, len(placed_logical)):
                    lq_a = placed_logical[i]
                    lq_b = placed_logical[j]
                    pq_a = perturbed_mapping[lq_a]
                    pq_b = perturbed_mapping[lq_b]

                    perturbed_mapping[lq_a] = pq_b
                    perturbed_mapping[lq_b] = pq_a
                    new_cost = compute_cost(perturbed_mapping)

                    if new_cost < p_cost:
                        perturbed_reverse[pq_a] = lq_b
                        perturbed_reverse[pq_b] = lq_a
                        p_cost = new_cost
                        improved = True
                    else:
                        perturbed_mapping[lq_a] = pq_a
                        perturbed_mapping[lq_b] = pq_b

        if p_cost < best_cost:
            best_cost = p_cost
            best_mapping = perturbed_mapping[:]
            best_reverse = perturbed_reverse[:]

    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        from src.utils.python_to_isl import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)