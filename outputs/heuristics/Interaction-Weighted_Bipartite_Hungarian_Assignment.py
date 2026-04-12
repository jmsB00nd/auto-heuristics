def init_mapping(self):
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # --- Step 1: Build weighted interaction graph from circuit gates ---
    # interaction_weight[(lq1, lq2)] = number of 2-qubit gates between lq1 and lq2
    interaction_weight = defaultdict(float)
    logical_qubit_set = set()

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback to trivial identity mapping if circuit has no 2-qubit gates
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 2: Compute weighted degree of each logical qubit ---
    # weighted_degree[lq] = total interaction weight summed over all of lq's partners
    weighted_degree = defaultdict(float)
    for (q1, q2), w in interaction_weight.items():
        weighted_degree[q1] += w
        weighted_degree[q2] += w

    # --- Step 3: Compute mean BFS distance from each physical qubit to all others ---
    # mean_dist[pq] serves as a centrality proxy: lower = more central in hardware graph
    mean_dist = {}
    for p in physical_qubits:
        finite_dists = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if o != p and self.distance_matrix[p][o] != float('inf')
        ]
        mean_dist[p] = sum(finite_dists) / len(finite_dists) if finite_dists else float('inf')

    n_lq = len(logical_qubits)
    n_pq = len(physical_qubits)

    # --- Step 4: Build cost matrix for the linear assignment problem ---
    # cost[i][j] = cost of assigning logical_qubits[i] to physical_qubits[j]
    #
    # Derivation: if lq is placed at pq, for each interaction partner lq2 with
    # weight w(lq, lq2), the routing distance to lq2 is unknown at assignment time.
    # We approximate it by the expected hardware distance from pq, which equals
    # the mean BFS distance from pq to all other physical qubits — capturing how
    # "isolated" pq is in the coupling graph. Summing over all partners:
    #
    #   cost(lq, pq) = sum_{lq2 in N(lq)} w(lq, lq2) * mean_dist(pq)
    #                = weighted_degree[lq] * mean_dist(pq)
    #
    # This naturally drives high-interaction logical qubits toward central physical
    # qubits (low mean_dist), minimizing expected routing overhead.
    cost_matrix = np.zeros((n_lq, n_pq))
    for i, lq in enumerate(logical_qubits):
        wd = weighted_degree[lq]
        for j, pq in enumerate(physical_qubits):
            cost_matrix[i, j] = wd * mean_dist[pq]

    # --- Step 5: Solve optimal linear assignment in O(n^3) via Hungarian algorithm ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # row_ind[k] -> index into logical_qubits, col_ind[k] -> index into physical_qubits
    lq_to_phys = {
        logical_qubits[r]: physical_qubits[c]
        for r, c in zip(row_ind, col_ind)
    }

    # --- Step 6: Build strict 1-to-1 bijection over all num_qubits indices ---
    # Start from identity permutation and apply assignments via in-place swaps,
    # ensuring every index in [0, num_qubits) is covered exactly once.
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        # Displace the logical qubit currently sitting at target_phys
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)