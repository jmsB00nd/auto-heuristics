def init_mapping(self):
    """Heat-Diffusion Mapping: degree matching + iterative local search + Hungarian."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    # ── 1. Collect logical qubits and interaction degrees ─────────────
    logical_qubits = sorted({q for qubits in self.access.values() for q in qubits})
    num_logical = len(logical_qubits)

    if num_logical == 0:
        self.mapping_dict = {}
        self.reverse_mapping_dict = {}
        return

    log_idx = {q: i for i, q in enumerate(logical_qubits)}

    # Logical interaction degree: number of 2-qubit gates involving each qubit
    d_L = defaultdict(int)
    # Neighbor sets for each logical qubit (other qubits it interacts with)
    logical_neighbors = defaultdict(set)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            d_L[q1] += 1
            d_L[q2] += 1
            logical_neighbors[q1].add(q2)
            logical_neighbors[q2].add(q1)

    # ── 2. Physical qubits and connectivity degrees ───────────────────
    phys_qubits = sorted(self.backend.keys())
    num_phys = len(phys_qubits)
    phys_idx = {q: i for i, q in enumerate(phys_qubits)}

    d_P = {q: len(self.backend[q]) for q in phys_qubits}

    # ── 3. Degree-sorted initial assignment ───────────────────────────
    # Sort logical qubits by interaction degree (descending)
    sorted_logical = sorted(logical_qubits, key=lambda q: d_L[q], reverse=True)
    # Sort physical qubits by connectivity degree (descending)
    sorted_physical = sorted(phys_qubits, key=lambda q: d_P[q], reverse=True)

    # Initial assignment: i-th busiest logical -> i-th most connected physical
    assignment = {}  # logical -> physical
    for i, lq in enumerate(sorted_logical):
        assignment[lq] = sorted_physical[i]

    # ── 4. Iterative local search refinement ──────────────────────────
    lam = 0.5  # lambda weight for proximity term

    def compute_total_cost(asgn):
        cost = 0.0
        for lq in logical_qubits:
            pq = asgn[lq]
            # Degree mismatch
            cost += abs(d_L[lq] - d_P[pq])
            # Proximity of interacting pairs
            for neighbor in logical_neighbors[lq]:
                if neighbor in asgn:
                    pn = asgn[neighbor]
                    cost += lam * self.distance_matrix[pq][pn]
        return cost

    current_cost = compute_total_cost(assignment)
    max_iterations = min(num_logical * num_logical, 500)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(num_logical):
            if improved:
                break
            for j in range(i + 1, num_logical):
                lq1 = sorted_logical[i]
                lq2 = sorted_logical[j]
                # Try swapping
                assignment[lq1], assignment[lq2] = assignment[lq2], assignment[lq1]
                new_cost = compute_total_cost(assignment)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                else:
                    # Revert
                    assignment[lq1], assignment[lq2] = assignment[lq2], assignment[lq1]

    # ── 5. Build refined cost matrix and apply Hungarian ──────────────
    cost_matrix = np.zeros((num_logical, num_phys))
    for i, lq in enumerate(logical_qubits):
        for j, pq in enumerate(phys_qubits):
            # Degree mismatch term
            deg_cost = abs(d_L[lq] - d_P[pq])
            # Proximity term: sum of distances to assigned neighbors
            prox_cost = 0.0
            for neighbor in logical_neighbors[lq]:
                if neighbor in assignment:
                    pn_idx = phys_idx[assignment[neighbor]]
                    prox_cost += self.distance_matrix[pq][phys_qubits[pn_idx]]
            cost_matrix[i, j] = deg_cost + lam * prox_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # ── 6. Populate mapping dicts ─────────────────────────────────────
    self.mapping_dict = {}
    self.reverse_mapping_dict = {}
    for li, pi in zip(row_ind, col_ind):
        lq = logical_qubits[li]
        pq = phys_qubits[pi]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)