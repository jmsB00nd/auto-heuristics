def init_mapping(self):
    """
    Spectral Flow Matching.

    Computes the Fiedler vector (second-smallest eigenvector of the Laplacian)
    for both the logical interaction graph and the hardware coupling graph.
    Sorts logical qubits by their Fiedler values and physical qubits by theirs,
    then matches them in sorted order. A local-search phase swaps pairs if doing
    so reduces total interaction-weighted distance.
    """
    import numpy as np
    import random
    from collections import defaultdict

    # --- Step 1: Collect logical qubits and build weighted interaction graph ---
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

    # Fallback: trivial identity if no logical qubits
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_lq = len(logical_qubits)
    lq_index = {lq: i for i, lq in enumerate(logical_qubits)}

    # --- Step 2: Compute Fiedler vector for the logical interaction graph ---
    L_logical = np.zeros((n_lq, n_lq))
    for (q1, q2), w in interaction_weight.items():
        i, j = lq_index[q1], lq_index[q2]
        L_logical[i, j] -= w
        L_logical[j, i] -= w
        L_logical[i, i] += w
        L_logical[j, j] += w

    eigenvalues_l, eigenvectors_l = np.linalg.eigh(L_logical)
    fiedler_col_l = 1 if n_lq > 1 else 0
    fiedler_logical = eigenvectors_l[:, fiedler_col_l]

    # Sort logical qubits by Fiedler value
    sorted_logical = [logical_qubits[i] for i in np.argsort(fiedler_logical)]

    # --- Step 3: Compute Fiedler vector for the hardware coupling graph ---
    n_pq = len(physical_qubits)
    pq_index = {pq: i for i, pq in enumerate(physical_qubits)}

    L_hardware = np.zeros((n_pq, n_pq))
    for pq in physical_qubits:
        for neighbor in self.backend[pq]:
            if neighbor in pq_index:
                i, j = pq_index[pq], pq_index[neighbor]
                if i < j:
                    L_hardware[i, j] -= 1
                    L_hardware[j, i] -= 1
                    L_hardware[i, i] += 1
                    L_hardware[j, j] += 1

    eigenvalues_h, eigenvectors_h = np.linalg.eigh(L_hardware)
    fiedler_col_h = 1 if n_pq > 1 else 0
    fiedler_hardware = eigenvectors_h[:, fiedler_col_h]

    # Sort physical qubits by Fiedler value
    sorted_physical = [physical_qubits[i] for i in np.argsort(fiedler_hardware)]

    # --- Step 4: Initial mapping by matching sorted positions ---
    lq_to_phys = {}
    for i, lq in enumerate(sorted_logical):
        if i < len(sorted_physical):
            lq_to_phys[lq] = sorted_physical[i]

    # Build strict 1-to-1 bijection via in-place swaps from identity
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    # --- Step 5: Local search refinement ---
    interaction_pairs = [(q1, q2, w) for (q1, q2), w in interaction_weight.items()]

    def total_cost(m):
        cost = 0
        for q1, q2, w in interaction_pairs:
            cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    current_cost = total_cost(mapping_dict)
    K = min(n_lq * n_lq * 5, 10000)

    for _ in range(K):
        a = random.choice(logical_qubits)
        b = random.choice(logical_qubits)
        if a == b:
            continue

        phys_a, phys_b = mapping_dict[a], mapping_dict[b]

        # Compute cost delta incrementally
        delta = 0
        for q1, q2, w in interaction_pairs:
            old_p1, old_p2 = mapping_dict[q1], mapping_dict[q2]
            new_p1 = phys_b if q1 == a else (phys_a if q1 == b else old_p1)
            new_p2 = phys_b if q2 == a else (phys_a if q2 == b else old_p2)
            delta += w * (self.distance_matrix[new_p1][new_p2] - self.distance_matrix[old_p1][old_p2])

        if delta < 0:
            mapping_dict[a] = phys_b
            mapping_dict[b] = phys_a
            reverse_mapping_dict[phys_a] = b
            reverse_mapping_dict[phys_b] = a
            current_cost += delta

    # --- Step 6: Finalize ---
    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)