def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

    # --- Step 1: Classical MDS to embed physical qubits in 2D ---
    D = np.zeros((n_phys, n_phys))
    for i, p1 in enumerate(physical_qubits):
        for j, p2 in enumerate(physical_qubits):
            D[i, j] = self.distance_matrix[p1][p2]

    # Classical MDS: double-center the squared distance matrix
    D_sq = D ** 2
    n = n_phys
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    # Eigen-decomposition, take top 2 components
    eigvals, eigvecs = np.linalg.eigh(B)
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    coords_2d = np.zeros((n, 2))
    for dim in range(min(2, n)):
        val = max(eigvals[dim], 0.0)
        coords_2d[:, dim] = eigvecs[:, dim] * np.sqrt(val)

    # Map physical qubit ID -> 2D position
    phys_pos = {}
    for i, pq in enumerate(physical_qubits):
        phys_pos[pq] = coords_2d[i]

    # --- Step 2: Build logical interaction graph from self.access ---
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0
            logical_degree[q1] += 1.0
            logical_degree[q2] += 1.0
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    logical_qubits = sorted(logical_qubits_set)

    # --- Step 3: Compute centroid of physical embedding ---
    centroid = np.mean(coords_2d, axis=0)

    # --- Step 4: Greedy gravity-based placement ---
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

    # Start with highest-degree logical qubit at physical qubit closest to centroid
    if logical_qubits:
        start_lq = max(logical_qubits, key=lambda q: logical_degree.get(q, 0))
        start_pq = nearest_free_physical(centroid)
        mapping_dict[start_lq] = start_pq
        reverse_mapping_dict[start_pq] = start_lq
        used_physical.add(start_pq)

        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            # Pick unplaced qubit with highest interaction to already-placed qubits
            best_lq = None
            best_weight = -1.0
            for lq in remaining:
                w = sum(logical_neighbors[lq].get(plq, 0.0) for plq in placed)
                if w > best_weight:
                    best_weight = w
                    best_lq = lq

            # Gravity center: weighted average position of placed neighbors
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

    # Fill remaining unmapped qubits with free physical qubits
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]

    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)