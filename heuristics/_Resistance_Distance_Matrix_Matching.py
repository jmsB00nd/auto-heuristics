def init_mapping(self):
    import numpy as np
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    # --- Step 1: Collect logical qubits and weighted interaction counts ---
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
    n_l = len(logical_qubits)
    n_p = len(physical_qubits)

    # Fallback: trivial identity mapping if circuit has no gates
    if n_l == 0:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    lq_idx = {q: i for i, q in enumerate(logical_qubits)}
    pq_idx = {p: i for i, p in enumerate(physical_qubits)}

    # --- Step 2: Build weighted circuit Laplacian L_c (n_l x n_l) ---
    L_c = np.zeros((n_l, n_l))
    for (q1, q2), w in interaction_weight.items():
        i, j = lq_idx[q1], lq_idx[q2]
        L_c[i, i] += w
        L_c[j, j] += w
        L_c[i, j] -= w
        L_c[j, i] -= w

    # --- Step 3: Resistance distances for circuit graph via Moore-Penrose pseudo-inverse ---
    L_c_pinv = np.linalg.pinv(L_c)
    diag_c = np.diag(L_c_pinv)
    R_c = diag_c[:, None] + diag_c[None, :] - 2.0 * L_c_pinv  # (n_l, n_l)
    np.fill_diagonal(R_c, 0.0)

    # --- Step 4: Build unweighted hardware Laplacian L_h (n_p x n_p) ---
    L_h = np.zeros((n_p, n_p))
    for p, neighbors in self.backend.items():
        pi = pq_idx[p]
        L_h[pi, pi] += len(neighbors)
        for nb in neighbors:
            nbi = pq_idx[nb]
            L_h[pi, nbi] -= 1.0

    # --- Step 5: Resistance distances for hardware graph ---
    L_h_pinv = np.linalg.pinv(L_h)
    diag_h = np.diag(L_h_pinv)
    R_h = diag_h[:, None] + diag_h[None, :] - 2.0 * L_h_pinv  # (n_p, n_p)
    np.fill_diagonal(R_h, 0.0)

    # --- Step 6: Build cost matrix via sorted resistance-profile matching ---
    # For each (logical i, physical p): compare the sorted resistance distance
    # profile of logical qubit i against the n_l nearest entries of physical qubit p.
    # Sorting gives a permutation-invariant structural fingerprint of the commute-time geometry.
    sorted_R_c = np.sort(R_c, axis=1)            # (n_l, n_l)
    sorted_R_h = np.sort(R_h, axis=1)[:, :n_l]  # (n_p, n_l) — nearest n_l entries

    # cost[i, p] = ||profile_circuit(i) - profile_hardware(p)||^2
    diff = sorted_R_c[:, None, :] - sorted_R_h[None, :, :]  # (n_l, n_p, n_l)
    cost = np.sum(diff ** 2, axis=2)  # (n_l, n_p)

    # --- Step 7: Optimal assignment via Hungarian algorithm ---
    row_ind, col_ind = linear_sum_assignment(cost)

    lq_to_phys = {
        logical_qubits[row_ind[k]]: physical_qubits[col_ind[k]]
        for k in range(len(row_ind))
    }

    # Fill any remaining unassigned logical qubits (isolated, no interactions)
    placed_phys = set(lq_to_phys.values())
    remaining_phys = [p for p in physical_qubits if p not in placed_phys]
    unassigned = [lq for lq in logical_qubits if lq not in lq_to_phys]
    for lq, phys in zip(unassigned, remaining_phys):
        lq_to_phys[lq] = phys

    # --- Step 8: Build strict 1-to-1 bijection over all num_qubits indices ---
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        # Displace the logical qubit currently occupying target_phys
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)