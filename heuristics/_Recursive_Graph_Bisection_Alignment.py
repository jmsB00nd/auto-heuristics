def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    # --- 1. Identify logical qubits actually used in the circuit ---
    used_logical = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            used_logical.add(q)
    used_logical = sorted(used_logical)
    num_logical = len(used_logical)

    # --- 2. Build temporally-weighted interaction graph ---
    interaction_weight = defaultdict(float)
    total_gates = len(self.access)
    for idx, (gate, qubits) in enumerate(self.access.items()):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            # Temporal decay: earlier gates matter more
            decay = 1.0 / (1.0 + idx / max(total_gates, 1))
            interaction_weight[key] += decay

    # --- 3. Build hardware adjacency for physical qubits ---
    all_physical = sorted(self.backend.keys())

    # --- 4. Spectral bisection helper using Fiedler vector ---
    def spectral_bisect(nodes, weight_func):
        """Bisect a set of nodes using the Fiedler vector of the weighted Laplacian."""
        n = len(nodes)
        if n <= 1:
            return nodes, []
        node_idx = {v: i for i, v in enumerate(nodes)}
        L = np.zeros((n, n), dtype=np.float64)
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j:
                    w = weight_func(u, v)
                    if w > 0:
                        L[i, j] = -w
                        L[j, i] = -w
                        L[i, i] += w
                        L[j, j] += w

        # Compute eigenvalues/vectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            # Fallback: split in half by order
            mid = n // 2
            return nodes[:mid], nodes[mid:]

        # Fiedler vector is the eigenvector for the second smallest eigenvalue
        # Find index of second smallest eigenvalue (first non-zero)
        sorted_indices = np.argsort(eigvals)
        fiedler_idx = sorted_indices[1] if n > 1 else 0
        fiedler = eigvecs[:, fiedler_idx]

        # Split by sign of Fiedler vector
        part_a = []
        part_b = []
        # Sort by fiedler value to get a balanced split
        order = np.argsort(fiedler)
        mid = n // 2
        for rank, idx in enumerate(order):
            if rank < mid:
                part_a.append(nodes[idx])
            else:
                part_b.append(nodes[idx])

        return part_a, part_b

    # --- 5. Hardware graph bisection ---
    def hw_bisect(phys_nodes, size_a, size_b):
        """Bisect physical nodes into groups of size_a and size_b using spectral method."""
        n = len(phys_nodes)
        if n <= 1:
            return phys_nodes, []

        def hw_weight(u, v):
            # Use inverse distance as weight (closer = stronger connection)
            d = self.distance_matrix[u][v]
            if d == 0 or d == float('inf'):
                return 0.0
            return 1.0 / d

        part_a, part_b = spectral_bisect(phys_nodes, hw_weight)

        # Rebalance to match desired sizes
        all_sorted = part_a + part_b
        # Re-split using spectral ordering but enforce sizes
        node_idx = {v: i for i, v in enumerate(phys_nodes)}
        n = len(phys_nodes)
        L = np.zeros((n, n), dtype=np.float64)
        for i, u in enumerate(phys_nodes):
            for j, v in enumerate(phys_nodes):
                if i < j:
                    w = hw_weight(u, v)
                    if w > 0:
                        L[i, j] = -w
                        L[j, i] = -w
                        L[i, i] += w
                        L[j, j] += w
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            sorted_indices = np.argsort(eigvals)
            fiedler_idx = sorted_indices[1] if n > 1 else 0
            fiedler = eigvecs[:, fiedler_idx]
            order = np.argsort(fiedler)
            part_a = [phys_nodes[order[i]] for i in range(size_a)]
            part_b = [phys_nodes[order[i]] for i in range(size_a, n)]
        except:
            part_a = phys_nodes[:size_a]
            part_b = phys_nodes[size_a:]

        return part_a, part_b

    # --- 6. Evaluate cross-partition cost ---
    def cross_cost(log_a, log_b, phys_a, phys_b):
        """Sum of interaction_weight * min cross-partition hardware distance."""
        cost = 0.0
        phys_a_set = set(phys_a)
        phys_b_set = set(phys_b)
        for (q1, q2), w in interaction_weight.items():
            if (q1 in set(log_a) and q2 in set(log_b)) or (q1 in set(log_b) and q2 in set(log_a)):
                # Cross-partition interaction: estimate min distance between partitions
                min_dist = float('inf')
                for pa in phys_a:
                    for pb in phys_b:
                        d = self.distance_matrix[pa][pb]
                        if d < min_dist:
                            min_dist = d
                cost += w * min_dist
        return cost

    # --- 7. Recursive bisection alignment ---
    def recursive_align(logical_nodes, physical_nodes):
        """Recursively bisect and align logical to physical partitions."""
        n = len(logical_nodes)
        if n == 0:
            return {}
        if n == 1:
            return {logical_nodes[0]: physical_nodes[0]}

        # Bisect logical qubits using interaction graph
        def log_weight(u, v):
            key = (min(u, v), max(u, v))
            return interaction_weight.get(key, 0.0)

        log_a, log_b = spectral_bisect(logical_nodes, log_weight)
        size_a = len(log_a)
        size_b = len(log_b)

        # Bisect physical qubits to match sizes
        phys_a, phys_b = hw_bisect(physical_nodes, size_a, size_b)

        # Try both assignments and pick the one with lower cross-partition cost
        cost_ab = cross_cost(log_a, log_b, phys_a, phys_b)
        cost_ba = cross_cost(log_a, log_b, phys_b, phys_a)

        if cost_ba < cost_ab:
            phys_a, phys_b = phys_b, phys_a

        # Recurse
        mapping = {}
        mapping.update(recursive_align(log_a, phys_a))
        mapping.update(recursive_align(log_b, phys_b))
        return mapping

    # --- 8. Run the recursive alignment ---
    # Select physical qubits to use (pick the most connected ones if more physical than logical)
    if num_logical <= len(all_physical):
        # Sort physical qubits by degree (most connected first)
        phys_by_degree = sorted(all_physical, key=lambda p: len(self.backend.get(p, set())), reverse=True)
        selected_physical = sorted(phys_by_degree[:num_logical])
    else:
        selected_physical = all_physical

    mapping_result = recursive_align(used_logical, selected_physical)

    # --- 9. Build full mapping arrays ---
    self.mapping_dict = list(range(self.num_qubits))
    self.reverse_mapping_dict = list(range(self.num_qubits))

    # Assign mapped qubits
    used_physical = set(mapping_result.values())
    unused_physical = [p for p in range(self.num_qubits) if p not in used_physical]
    unmapped_logical = [q for q in range(self.num_qubits) if q not in used_logical]

    for lq, pq in mapping_result.items():
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    # Map remaining unmapped logical qubits to remaining physical qubits
    for lq, pq in zip(unmapped_logical, unused_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)