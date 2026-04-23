def init_mapping(self):
    import numpy as np

    n = self.num_qubits

    # --- Logical interaction graph Laplacian ---
    logical_qubits = set()
    for q1, neighbors in self.qubit_interaction_graph.items():
        logical_qubits.add(q1)
        for q2 in neighbors:
            logical_qubits.add(q2)
    logical_qubits = sorted(logical_qubits)
    num_logical = len(logical_qubits)

    if num_logical < 2:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Build weighted Laplacian for logical interaction graph
    lq_to_idx = {q: i for i, q in enumerate(logical_qubits)}
    L_logical = np.zeros((num_logical, num_logical))
    for q1, neighbors in self.qubit_interaction_graph.items():
        i = lq_to_idx[q1]
        for q2, weight in neighbors.items():
            j = lq_to_idx[q2]
            if i != j:
                L_logical[i, j] -= weight
                L_logical[i, i] += weight

    # Symmetrize (already symmetric from build, but ensure numerical symmetry)
    L_logical = (L_logical + L_logical.T) / 2.0

    # Fiedler vector for logical graph
    eigvals_l, eigvecs_l = np.linalg.eigh(L_logical)
    fiedler_logical = eigvecs_l[:, 1]

    # --- Physical coupling graph Laplacian ---
    physical_qubits = sorted(self.backend.keys())
    num_physical = len(physical_qubits)
    pq_to_idx = {q: i for i, q in enumerate(physical_qubits)}
    L_physical = np.zeros((num_physical, num_physical))
    for q1, neighbors in self.backend.items():
        i = pq_to_idx[q1]
        for q2 in neighbors:
            j = pq_to_idx[q2]
            if i != j:
                L_physical[i, j] -= 1
                L_physical[i, i] += 1

    L_physical = (L_physical + L_physical.T) / 2.0

    eigvals_p, eigvecs_p = np.linalg.eigh(L_physical)
    fiedler_physical = eigvecs_p[:, 1]

    # --- Rank-match by Fiedler vector components ---
    # Sort logical qubits by their Fiedler component
    logical_order = sorted(range(num_logical), key=lambda i: fiedler_logical[i])
    sorted_logical = [logical_qubits[i] for i in logical_order]

    # Sort physical qubits by their Fiedler component
    physical_order = sorted(range(num_physical), key=lambda i: fiedler_physical[i])
    sorted_physical = [physical_qubits[i] for i in physical_order]

    # Assign: k-th logical -> k-th physical
    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))
    used_physical = set()

    num_to_match = min(num_logical, num_physical)
    for k in range(num_to_match):
        lq = sorted_logical[k]
        pq = sorted_physical[k]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)

    # Assign remaining unmapped logical qubits to leftover physical qubits
    all_physical = set(range(n))
    remaining_physical = sorted(all_physical - used_physical)
    mapped_logical = set(sorted_logical[:num_to_match])
    unmapped_logical = [q for q in range(n) if q not in mapped_logical]

    for lq, pq in zip(unmapped_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)