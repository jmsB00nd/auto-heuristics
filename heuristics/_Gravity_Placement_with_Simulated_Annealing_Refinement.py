def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    import random
    import math

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

    # ---------------------------------------------------------------
    # Phase 1a: Classical MDS to embed physical qubits in 2D
    # ---------------------------------------------------------------
    D = np.zeros((n_phys, n_phys))
    for i, p1 in enumerate(physical_qubits):
        for j, p2 in enumerate(physical_qubits):
            D[i, j] = self.distance_matrix[p1][p2]

    D_sq = D ** 2
    n = n_phys
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    coords_2d = np.zeros((n, 2))
    for dim in range(min(2, n)):
        val = max(eigvals[dim], 0.0)
        coords_2d[:, dim] = eigvecs[:, dim] * np.sqrt(val)

    phys_pos = {}
    for i, pq in enumerate(physical_qubits):
        phys_pos[pq] = coords_2d[i]

    # ---------------------------------------------------------------
    # Phase 1b: Build DAG and compute topological rank for temporal decay
    # ---------------------------------------------------------------
    schedule = sorted(self.access.keys())

    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in schedule:
        write_qubits = self.write_dict.get(node, [])
        read_qubits = [q for q in self.access[node] if q not in write_qubits]

        for q in read_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            active_readers[q].add(node)

        for q in write_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            for r in active_readers.get(q, set()):
                if r != node:
                    successors[r].add(node)
                    predecessors[node].add(r)
            active_readers[q].clear()
            latest_writer[q] = node

    # Kahn's topological sort
    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_rank = {}
    rank = 0
    while queue:
        g = queue.popleft()
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    # ---------------------------------------------------------------
    # Phase 1c: Build temporal-decay weighted interaction graph
    # ---------------------------------------------------------------
    alpha = 2.5
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha * r / total_gates)
            interaction_weight[key] += w
            logical_degree[q1] += w
            logical_degree[q2] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    logical_qubits = sorted(logical_qubits_set)

    # ---------------------------------------------------------------
    # Phase 1d: Greedy gravity-based placement (MDS centroid pull)
    # ---------------------------------------------------------------
    centroid = np.mean(coords_2d, axis=0)
    used_physical = set()
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q

    def nearest_free_physical(target_pos):
        best_pq = None
        best_dist = float('inf')
        for pq in physical_qubits:
            if pq not in used_physical:
                d = np.linalg.norm(phys_pos[pq] - target_pos)
                if d < best_dist:
                    best_dist = d
                    best_pq = pq
        return best_pq

    if logical_qubits:
        start_lq = max(logical_qubits, key=lambda q: logical_degree.get(q, 0))
        start_pq = nearest_free_physical(centroid)
        mapping_dict[start_lq] = start_pq
        reverse_mapping_dict[start_pq] = start_lq
        used_physical.add(start_pq)

        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            best_lq = None
            best_weight = -1.0
            for lq in remaining:
                w = sum(logical_neighbors[lq].get(plq, 0.0) for plq in placed)
                if w > best_weight:
                    best_weight = w
                    best_lq = lq

            neighbors_placed = {plq: logical_neighbors[best_lq].get(plq, 0.0)
                                for plq in placed if plq in logical_neighbors[best_lq]}

            if neighbors_placed:
                total_w = sum(neighbors_placed.values())
                gravity = np.zeros(2)
                for plq, w in neighbors_placed.items():
                    gravity += w * phys_pos[mapping_dict[plq]]
                gravity /= total_w
            else:
                gravity = centroid

            pq = nearest_free_physical(gravity)
            mapping_dict[best_lq] = pq
            reverse_mapping_dict[pq] = best_lq
            used_physical.add(pq)
            placed.add(best_lq)
            remaining.discard(best_lq)

    # Fill remaining unmapped qubits
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    # ---------------------------------------------------------------
    # Phase 2: Simulated Annealing Refinement
    # ---------------------------------------------------------------
    # Precompute list of interacting pairs for cost function
    interaction_pairs = list(interaction_weight.keys())

    def compute_cost(m_dict):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            cost += w * self.distance_matrix[m_dict[q1]][m_dict[q2]]
        return cost

    # Only run SA if there are interactions to optimize
    if interaction_pairs:
        interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

        if len(interacting_logical) >= 2:
            current_cost = compute_cost(mapping_dict)

            # Calibrate T0 to 10% of initial cost
            T0 = 0.1 * current_cost if current_cost > 0 else 1.0
            beta = 0.995
            num_iterations = 2000
            T = T0

            for iteration in range(num_iterations):
                # Pick two random logical qubits to swap
                idx_a = random.randint(0, len(interacting_logical) - 1)
                idx_b = random.randint(0, len(interacting_logical) - 2)
                if idx_b >= idx_a:
                    idx_b += 1
                lq_a = interacting_logical[idx_a]
                lq_b = interacting_logical[idx_b]
                pq_a = mapping_dict[lq_a]
                pq_b = mapping_dict[lq_b]

                # Compute delta cost efficiently (only edges incident to lq_a or lq_b)
                delta = 0.0
                affected_qubits = set()
                affected_qubits.update(logical_neighbors[lq_a].keys())
                affected_qubits.update(logical_neighbors[lq_b].keys())

                for q in affected_qubits:
                    if q == lq_a or q == lq_b:
                        continue
                    pq_q = mapping_dict[q]

                    w_a = logical_neighbors[lq_a].get(q, 0.0)
                    if w_a > 0:
                        delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])

                    w_b = logical_neighbors[lq_b].get(q, 0.0)
                    if w_b > 0:
                        delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

                # Handle the direct edge between lq_a and lq_b (distance unchanged by swap)
                # d(pq_b, pq_a) == d(pq_a, pq_b), so no delta contribution

                # SA acceptance criterion
                if delta < 0:
                    # Accept improving move
                    mapping_dict[lq_a] = pq_b
                    mapping_dict[lq_b] = pq_a
                    reverse_mapping_dict[pq_a] = lq_b
                    reverse_mapping_dict[pq_b] = lq_a
                    current_cost += delta
                elif T > 1e-12:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        mapping_dict[lq_a] = pq_b
                        mapping_dict[lq_b] = pq_a
                        reverse_mapping_dict[pq_a] = lq_b
                        reverse_mapping_dict[pq_b] = lq_a
                        current_cost += delta

                # Geometric cooling
                T *= beta

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)