def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    num_q = self.num_qubits
    
    # --- Step 0: Identify logical qubits and 2-qubit gates ---
    logical_qubit_set = set()
    two_qubit_gates = []
    gate_order = sorted(self.access.keys())  # deterministic gate ordering
    
    for gate in gate_order:
        qubits = self.access[gate]
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            two_qubit_gates.append(gate)
    
    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())
    
    # --- Step 1: Partition into K time windows ---
    K = min(4, max(1, len(two_qubit_gates) // 5 + 1))  # 1-4 windows
    if len(two_qubit_gates) == 0:
        K = 1
    
    windows = []
    if K > 0 and len(two_qubit_gates) > 0:
        chunk_size = max(1, len(two_qubit_gates) // K)
        for i in range(K):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < K - 1 else len(two_qubit_gates)
            windows.append(two_qubit_gates[start:end])
    else:
        windows = [two_qubit_gates]
    
    # --- Step 2: Build per-window interaction graphs ---
    window_interactions = []
    for window in windows:
        interaction = defaultdict(float)
        for gate in window:
            qubits = self.access[gate]
            if len(qubits) == 2:
                q1, q2 = qubits
                pair = (min(q1, q2), max(q1, q2))
                interaction[pair] += 1.0
        window_interactions.append(interaction)
    
    # --- Step 3: MDS embedding of hardware graph ---
    # Use classical MDS on the distance matrix
    n_phys = len(physical_qubits)
    
    # Build distance submatrix for physical qubits
    D = np.zeros((n_phys, n_phys))
    for i, p1 in enumerate(physical_qubits):
        for j, p2 in enumerate(physical_qubits):
            D[i][j] = self.distance_matrix[p1][p2]
    
    # Classical MDS to get 2D coordinates
    D_sq = D ** 2
    n = n_phys
    H = np.eye(n) - np.ones((n, n)) / n  # centering matrix
    B = -0.5 * H @ D_sq @ H
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Take top 2 positive eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    dim = 2
    pos_mask = eigenvalues[:dim] > 0
    coords_phys = np.zeros((n, dim))
    for d in range(dim):
        if d < len(eigenvalues) and eigenvalues[d] > 0:
            coords_phys[:, d] = eigenvectors[:, d] * np.sqrt(eigenvalues[d])
    
    # Map physical qubit index to MDS coordinate
    phys_to_idx = {p: i for i, p in enumerate(physical_qubits)}
    
    # --- Step 4: Gravity placement using Window-1 interaction graph ---
    interaction_w1 = window_interactions[0] if window_interactions else {}
    
    # Compute logical qubit "mass" (total interaction weight) from window 1
    lq_mass = defaultdict(float)
    for (q1, q2), w in interaction_w1.items():
        lq_mass[q1] += w
        lq_mass[q2] += w
    
    # Sort logical qubits by mass descending (heaviest first)
    sorted_lq = sorted(logical_qubits, key=lambda q: lq_mass.get(q, 0), reverse=True)
    
    # Gravity placement: place each logical qubit at the physical position
    # that minimizes weighted distance to already-placed neighbors
    occupied = set()
    lq_to_phys = {}
    
    for lq in sorted_lq:
        if not occupied:
            # Place first qubit at most central physical qubit
            centrality = []
            for i, p in enumerate(physical_qubits):
                avg_dist = np.mean(D[i])
                centrality.append((avg_dist, p))
            centrality.sort()
            best_p = centrality[0][1]
            lq_to_phys[lq] = best_p
            occupied.add(best_p)
            continue
        
        # Compute gravity pull toward already-placed neighbors
        best_score = float('inf')
        best_p = None
        
        for p in physical_qubits:
            if p in occupied:
                continue
            p_idx = phys_to_idx[p]
            score = 0.0
            for placed_lq, placed_p in lq_to_phys.items():
                pair = (min(lq, placed_lq), max(lq, placed_lq))
                w = interaction_w1.get(pair, 0)
                if w > 0:
                    score += w * D[p_idx][phys_to_idx[placed_p]]
            best_score_candidate = score
            if best_score_candidate < best_score:
                best_score = best_score_candidate
                best_p = p
        
        if best_p is None:
            # No unoccupied qubit found (shouldn't happen), pick any free one
            for p in physical_qubits:
                if p not in occupied:
                    best_p = p
                    break
        
        lq_to_phys[lq] = best_p
        occupied.add(best_p)
    
    # --- Step 5: Multi-window cost evaluation function ---
    def compute_multi_window_cost(mapping):
        """Compute weighted cost across all windows."""
        total = 0.0
        for w_idx, interaction in enumerate(window_interactions):
            weight = 0.5 ** w_idx  # 1, 0.5, 0.25, ...
            window_cost = 0.0
            for (q1, q2), freq in interaction.items():
                p1 = mapping[q1]
                p2 = mapping[q2]
                window_cost += freq * self.distance_matrix[p1][p2]
            total += weight * window_cost
        return total
    
    # --- Step 6: Build full mapping from gravity placement ---
    # Start with identity mapping
    mapping_dict = list(range(num_q))
    reverse_mapping_dict = list(range(num_q))
    
    # Apply gravity placement assignments
    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq
    
    # --- Step 7: Constrained 2-opt improvement ---
    if len(window_interactions) > 1 and len(logical_qubits) > 1:
        current_cost = compute_multi_window_cost(mapping_dict)
        
        improved = True
        max_iterations = 3  # limit iterations for efficiency
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for i in range(len(logical_qubits)):
                for j in range(i + 1, len(logical_qubits)):
                    lq_i = logical_qubits[i]
                    lq_j = logical_qubits[j]
                    p_i = mapping_dict[lq_i]
                    p_j = mapping_dict[lq_j]
                    
                    # Try swap
                    mapping_dict[lq_i] = p_j
                    mapping_dict[lq_j] = p_i
                    reverse_mapping_dict[p_j] = lq_i
                    reverse_mapping_dict[p_i] = lq_j
                    
                    new_cost = compute_multi_window_cost(mapping_dict)
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                    else:
                        # Revert swap
                        mapping_dict[lq_i] = p_i
                        mapping_dict[lq_j] = p_j
                        reverse_mapping_dict[p_i] = lq_i
                        reverse_mapping_dict[p_j] = lq_j
    
    # --- Final: Set mappings ---
    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    
    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)