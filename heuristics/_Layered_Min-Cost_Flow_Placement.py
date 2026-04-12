def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    num_q = self.num_qubits

    # --- Step 1: Identify logical qubits used in 2-qubit gates and build interaction weights ---
    interaction_weight = defaultdict(float)
    logical_qubits_used = set()
    
    # Build a simple topological ordering proxy: gate index as temporal position
    gate_ids = sorted(self.access.keys())
    total_gates = len(gate_ids) if gate_ids else 1
    
    decay_factor = 0.9  # temporal decay
    
    for idx, gate_id in enumerate(gate_ids):
        qubits = self.access[gate_id]
        for q in qubits:
            logical_qubits_used.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            # Temporal decay: earlier gates matter more
            weight = decay_factor ** (idx / total_gates * 10)
            pair = (min(q1, q2), max(q1, q2))
            interaction_weight[pair] += weight

    logical_qubits_list = sorted(logical_qubits_used)
    num_logical = len(logical_qubits_list)
    
    if num_logical == 0:
        # No qubits used, trivial mapping
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        self.mapping = list(range(num_q))
        self.reverse_mapping = list(range(num_q))
        if self.use_isl:
            from src.utils.isl_data_loader import dict_to_isl_map
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Map logical qubit IDs to indices for the cost matrix
    lq_to_idx = {lq: i for i, lq in enumerate(logical_qubits_list)}
    
    # Build interaction profile for each logical qubit
    # interaction_profile[i][j] = total weighted interaction between logical_qubits_list[i] and logical_qubits_list[j]
    interaction_matrix = np.zeros((num_logical, num_logical))
    for (q1, q2), w in interaction_weight.items():
        if q1 in lq_to_idx and q2 in lq_to_idx:
            i, j = lq_to_idx[q1], lq_to_idx[q2]
            interaction_matrix[i][j] = w
            interaction_matrix[j][i] = w

    # Physical qubits: 0 to num_q-1 (but distance_matrix may be sized differently)
    num_physical = len(self.distance_matrix)
    physical_qubits_list = list(range(num_physical))

    # --- Step 2: Compute average neighborhood distance for each physical qubit ---
    avg_neighbor_dist = np.zeros(num_physical)
    for p in range(num_physical):
        neighbors = self.backend.get(p, set())
        if neighbors:
            avg_neighbor_dist[p] = np.mean([self.distance_matrix[p][n] for n in neighbors])
        else:
            avg_neighbor_dist[p] = float('inf')

    # --- Step 3: Build initial cost matrix using relaxation ---
    # Cost of assigning logical qubit l (index i) to physical qubit p:
    # sum over all partners l' of l: w(l,l') * avg_neighbor_dist(p)
    # This is a relaxation since we don't know where l' will be placed yet
    
    cost_matrix = np.zeros((num_logical, num_physical))
    
    # Interaction strength per logical qubit
    interaction_sum = interaction_matrix.sum(axis=1)  # total interaction weight for each logical qubit
    
    for i in range(num_logical):
        for p in range(num_physical):
            cost_matrix[i][p] = interaction_sum[i] * avg_neighbor_dist[p]

    # --- Step 4: Iterative refinement (2-3 passes) ---
    num_iterations = 3
    assignment = None
    
    for iteration in range(num_iterations):
        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment = col_ind  # assignment[i] = physical qubit for logical qubit index i
        
        if iteration < num_iterations - 1:
            # Recompute cost matrix using actual assigned positions
            cost_matrix = np.zeros((num_logical, num_physical))
            
            for i in range(num_logical):
                for p in range(num_physical):
                    cost = 0.0
                    for j in range(num_logical):
                        if interaction_matrix[i][j] > 0:
                            # Use actual assigned position of partner j from previous iteration
                            p_partner = assignment[j]
                            cost += interaction_matrix[i][j] * self.distance_matrix[p][p_partner]
                    cost_matrix[i][p] = cost

    # --- Step 5: Build the mapping ---
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q
    
    used_physical = set()
    for i, lq in enumerate(logical_qubits_list):
        pq = assignment[i]
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq
        used_physical.add(pq)

    # Assign remaining logical qubits (not in any gate) to free physical qubits
    free_physical = [p for p in range(num_physical) if p not in used_physical]
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    self.mapping = list(mapping_dict)
    self.reverse_mapping = list(reverse_mapping_dict)
    
    if self.use_isl:
        from src.utils.isl_data_loader import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)