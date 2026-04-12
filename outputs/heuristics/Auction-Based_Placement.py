def init_mapping(self):
    from collections import defaultdict

    # --- Step 1: Extract logical qubits and build interaction graph ---
    logical_qubit_set = set()
    interaction_weight = defaultdict(float)

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    if not logical_qubit_set:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())
    n_logical = len(logical_qubits)
    n_physical = len(physical_qubits)

    lq_to_idx = {lq: i for i, lq in enumerate(logical_qubits)}
    pq_to_idx = {pq: j for j, pq in enumerate(physical_qubits)}

    # --- Step 2: Build interaction neighbor lists ---
    lq_neighbors = defaultdict(list)
    for (q1, q2), w in interaction_weight.items():
        lq_neighbors[q1].append((q2, w))
        lq_neighbors[q2].append((q1, w))

    # --- Step 3: Compute benefit matrix for the auction ---
    # Physical qubit centrality: sum of distances to all other physical qubits
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in physical_qubits)

    # Logical qubit total interaction weight
    lq_total_weight = {}
    for lq in logical_qubits:
        lq_total_weight[lq] = sum(w for _, w in lq_neighbors.get(lq, []))

    # benefit[i][j] = -(total_interaction_weight[i] * centrality[j])
    # High-interaction logical qubits prefer central physical qubits;
    # the auction algorithm resolves contention optimally.
    benefit = [[0.0] * n_physical for _ in range(n_logical)]
    for lq in logical_qubits:
        i = lq_to_idx[lq]
        tw = lq_total_weight.get(lq, 0.0)
        for j, pq in enumerate(physical_qubits):
            benefit[i][j] = -(tw * phys_centrality[pq])

    # --- Step 4: Bertsekas Auction Algorithm ---
    # Bidders = logical qubits, Items = physical qubits
    # Handles asymmetry (n_physical >= n_logical) naturally.
    epsilon = 1.0 / (n_logical + 1)
    prices = [0.0] * n_physical
    assignment = [-1] * n_logical
    assigned_to = [-1] * n_physical

    unassigned = set(range(n_logical))
    max_iterations = n_logical * n_physical + n_logical * 10

    iteration = 0
    while unassigned and iteration < max_iterations:
        iteration += 1
        i = next(iter(unassigned))

        # Find best and second-best (value - price) for bidder i
        best_j = -1
        best_value = -float('inf')
        second_best_value = -float('inf')

        for j in range(n_physical):
            v = benefit[i][j] - prices[j]
            if v > best_value:
                second_best_value = best_value
                best_value = v
                best_j = j
            elif v > second_best_value:
                second_best_value = v

        # Bid: raise price of best item
        bid_increment = best_value - second_best_value + epsilon
        prices[best_j] += bid_increment

        # Unassign previous owner if any
        prev_owner = assigned_to[best_j]
        if prev_owner != -1:
            assignment[prev_owner] = -1
            unassigned.add(prev_owner)

        # Assign bidder i to item best_j
        assignment[i] = best_j
        assigned_to[best_j] = i
        unassigned.discard(i)

    # --- Step 5: Build mapping from auction result ---
    lq_to_phys = {}
    used_phys = set()
    for i, lq in enumerate(logical_qubits):
        j = assignment[i]
        if j != -1:
            pq = physical_qubits[j]
            lq_to_phys[lq] = pq
            used_phys.add(pq)
        else:
            # Fallback for unassigned (should not happen)
            for pq in physical_qubits:
                if pq not in used_phys:
                    lq_to_phys[lq] = pq
                    used_phys.add(pq)
                    break

    # --- Step 6: Convert to strict bijection via swap-in-place ---
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

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)