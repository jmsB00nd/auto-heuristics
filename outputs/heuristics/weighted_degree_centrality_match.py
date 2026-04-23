def init_mapping(self):
    num_q = self.num_qubits

    # Gather activity scores for logical qubits involved in 2-qubit gates
    active_logical = sorted(
        self.logical_activity.keys(),
        key=lambda q: self.logical_activity[q],
        reverse=True,
    )

    # Sort physical qubits by descending closeness centrality
    all_physical = list(range(num_q))
    sorted_physical = sorted(
        all_physical,
        key=lambda p: self.physical_centrality.get(p, 0),
        reverse=True,
    )

    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    used_physical = set()
    assigned_logical = set()

    phys_iter = iter(sorted_physical)

    for lq in active_logical:
        if lq >= num_q:
            continue
        pq = next(phys_iter)
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        assigned_logical.add(lq)

    # Remaining physical qubits (in centrality order) for unassigned logical qubits
    remaining_physical = [p for p in sorted_physical if p not in used_physical]
    remaining_logical = [q for q in range(num_q) if q not in assigned_logical]

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)