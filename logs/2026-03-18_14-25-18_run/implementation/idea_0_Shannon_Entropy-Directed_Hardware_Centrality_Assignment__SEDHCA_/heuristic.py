def init_mapping(self):
    """
    Shannon Entropy-Directed Hardware Centrality Assignment (SEDHCA).

    Core idea (information-theoretic):
      - A logical qubit with HIGH interaction entropy (spreads interactions
        broadly over many partners) needs routing flexibility → place it on
        a CENTRAL physical qubit (small mean BFS distance to all others).
      - A logical qubit with LOW entropy (few, focused partners) tolerates
        a peripheral physical qubit.

    Steps:
      1. Build interaction-frequency distributions per logical qubit from
         the 2-qubit gates in self.access.
      2. Compute Shannon entropy H(q) = -Σ p log₂ p for each logical qubit.
      3. Compute closeness centrality C(p) = (|P|-1) / Σ_j dist(p,j) for
         each physical qubit using the precomputed distance_matrix.
      4. Sort logical qubits by H descending; sort physical qubits by C
         descending; assign in rank order (greedy bijection).
      5. Populate self.mapping_dict and self.reverse_mapping_dict (lists).
    """
    # ── 1. Collect all logical qubits that appear in the circuit ──────────
    logical_qubits_set = set()
    for qubits in self.access.values():
        for q in qubits:
            logical_qubits_set.add(q)
    logical_qubits = sorted(logical_qubits_set)

    # ── 2. Interaction-frequency distribution for each logical qubit ──────
    # interaction_counts[q][partner] = number of 2-qubit gates between them
    interaction_counts = defaultdict(lambda: defaultdict(int))
    for qubits in self.access.values():
        if len(qubits) == 2:
            q1, q2 = qubits
            interaction_counts[q1][q2] += 1
            interaction_counts[q2][q1] += 1

    # ── 3. Shannon entropy H(q) for each logical qubit ────────────────────
    def _shannon_entropy(freq_dict):
        total = sum(freq_dict.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in freq_dict.values():
            p = count / total
            if p > 0.0:
                entropy -= p * math.log2(p)
        return entropy

    logical_entropy = {q: _shannon_entropy(interaction_counts[q])
                       for q in logical_qubits}

    # ── 4. Closeness centrality C(p) for each physical qubit ──────────────
    # Physical qubits = nodes present in the backend adjacency list.
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

    closeness = {}
    for pq in physical_qubits:
        # Sum of BFS distances to every other physical qubit
        total_dist = sum(
            self.distance_matrix[pq][other]
            for other in physical_qubits
            if other != pq and self.distance_matrix[pq][other] != float('inf')
        )
        if total_dist > 0 and n_phys > 1:
            closeness[pq] = (n_phys - 1) / total_dist
        else:
            # Isolated node or single-node graph: assign lowest centrality
            closeness[pq] = 0.0

    # ── 5. Rank-ordered greedy bijection ──────────────────────────────────
    # High-entropy logical → high-centrality physical
    sorted_logical = sorted(logical_qubits,
                            key=lambda q: logical_entropy[q],
                            reverse=True)
    sorted_physical = sorted(physical_qubits,
                             key=lambda p: closeness[p],
                             reverse=True)

    # ── 6. Populate mapping lists ─────────────────────────────────────────
    self.mapping_dict = [-1] * self.num_qubits
    self.reverse_mapping_dict = [-1] * self.num_qubits

    for rank, lq in enumerate(sorted_logical):
        pq = sorted_physical[rank]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    # Keep list-based aliases consistent (used elsewhere in the class)
    self.mapping = self.mapping_dict[:]
    self.reverse_mapping = self.reverse_mapping_dict[:]

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)