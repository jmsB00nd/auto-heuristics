def init_mapping(self):
    import numpy as np
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    # --- Step 1: Collect logical qubits and physical qubits ---
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback to trivial identity if circuit has no gates
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_l = len(logical_qubits)
    n_p = len(physical_qubits)

    # --- Step 2: Build DAG and compute per-gate dependency counts ---
    # Uses the same pipeline as run(), giving us a DAG weight for each 2-qubit gate.
    successors2q, dag_preds2q, _, _, access2q = generate_dag(
        self.access, self.write_dict, self.num_qubits,
        enforce_read_after_read=True, transitive_reduction=True
    )
    dag_dep_count = compute_transitive_closure_bitset(successors2q, dag_preds2q)

    # --- Step 3: Compute DAG-weighted interaction load per logical qubit ---
    # For each 2-qubit gate g involving qubit l, contribute (dag_dep_count[g] + 1)
    # to l's total interaction weight.  The +1 ensures non-zero weight for sink gates.
    interaction_weight = defaultdict(float)
    for gate, qubits in access2q.items():
        if len(qubits) == 2:
            l1, l2 = qubits
            w = dag_dep_count[gate] + 1
            interaction_weight[l1] += w
            interaction_weight[l2] += w

    # --- Step 4: Compute mean BFS distance from each physical qubit to all others ---
    # mean_dist[p] captures centrality: a low value means p is close to the rest of
    # the hardware graph, making it a good host for a highly-interacting logical qubit.
    dm = self.distance_matrix
    mean_dist = []
    for p in physical_qubits:
        finite = [dm[p][o] for o in physical_qubits if o != p and dm[p][o] != float('inf')]
        mean_dist.append(sum(finite) / len(finite) if finite else float('inf'))

    # --- Step 5: Build cost matrix M[i][j] ---
    # Under the uniform-random-partner approximation, the expected routing cost of
    # placing logical qubit l at physical qubit p is:
    #   cost(l, p) = sum_{gates g involving l} (dag_dep[g]+1) * E[dist(p, partner)]
    #             = interaction_weight[l] * mean_dist[p]
    # This is the linear model solved optimally by the Hungarian algorithm.
    M = np.empty((n_l, n_p))
    for i, l in enumerate(logical_qubits):
        w_l = interaction_weight[l]
        for j in range(n_p):
            M[i, j] = w_l * mean_dist[j]

    # --- Step 6: Globally optimal assignment under the linear cost model ---
    row_ind, col_ind = linear_sum_assignment(M)

    lq_to_phys = {
        logical_qubits[row_ind[k]]: physical_qubits[col_ind[k]]
        for k in range(len(row_ind))
    }

    # Assign any isolated logical qubits (zero interaction weight, not in access2q)
    placed_phys = set(lq_to_phys.values())
    remaining_phys = [p for p in physical_qubits if p not in placed_phys]
    for lq, phys in zip([lq for lq in logical_qubits if lq not in lq_to_phys], remaining_phys):
        lq_to_phys[lq] = phys

    # --- Step 7: Build strict 1-to-1 bijection over all num_qubits indices ---
    # Start from the identity and use displacement to avoid collisions.
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        # Displace whoever currently occupies target_phys
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)