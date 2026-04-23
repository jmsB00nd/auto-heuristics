def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    N = self.num_qubits

    if self.access2q is None:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Collect interaction partners and weights per logical qubit
    # interaction_partners[l] = {q': total_weight}
    interaction_partners = defaultdict(lambda: defaultdict(float))

    # Compute DAG depth for each gate (longest path from a root)
    gate_depth = {}
    from collections import deque

    # Build predecessor counts for 2q gates
    in_degree = defaultdict(int)
    all_2q_gates = set(self.access2q.keys())
    successors = {}
    predecessors = {}

    # Use dag2q and dag_predecessors2q if available, else approximate
    if hasattr(self, 'dag2q') and self.dag2q is not None:
        for g in all_2q_gates:
            successors[g] = self.dag2q.get(g, set())
            predecessors[g] = self.dag_predecessors2q.get(g, set())
    else:
        # No DAG built yet; assign uniform depth
        for g in all_2q_gates:
            gate_depth[g] = 0
        successors = {g: set() for g in all_2q_gates}
        predecessors = {g: set() for g in all_2q_gates}

    if not gate_depth:
        # BFS topological order to compute depth
        for g in all_2q_gates:
            in_degree[g] = len(predecessors.get(g, set()) & all_2q_gates)

        queue = deque()
        for g in all_2q_gates:
            if in_degree[g] == 0:
                gate_depth[g] = 0
                queue.append(g)

        max_depth = 0
        while queue:
            g = queue.popleft()
            d = gate_depth[g]
            for s in successors.get(g, set()):
                if s not in all_2q_gates:
                    continue
                new_d = d + 1
                if s not in gate_depth or new_d > gate_depth[s]:
                    gate_depth[s] = new_d
                    if new_d > max_depth:
                        max_depth = new_d
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)

        # Gates not reached get depth 0
        for g in all_2q_gates:
            if g not in gate_depth:
                gate_depth[g] = 0
        max_depth = max(gate_depth.values()) if gate_depth else 0
    else:
        max_depth = max(gate_depth.values()) if gate_depth else 0

    # Build interaction partners with inverse-depth weighting
    for g in all_2q_gates:
        q1, q2 = self.access2q[g]
        depth = gate_depth.get(g, 0)
        weight = 1.0 / (1.0 + depth)
        interaction_partners[q1][q2] += weight
        interaction_partners[q2][q1] += weight

    # Identify logical qubits involved in 2q gates
    logical_qubits_2q = sorted(interaction_partners.keys())

    # Degree of each logical qubit (sum of interaction weights)
    logical_degree = {}
    for l in logical_qubits_2q:
        logical_degree[l] = sum(interaction_partners[l].values())

    # Sort logical qubits by degree descending
    logical_sorted = sorted(logical_qubits_2q, key=lambda l: logical_degree[l], reverse=True)

    # Compute physical qubit connectivity degree
    physical_degree = [0] * N
    for p in range(N):
        if p in self.backend:
            physical_degree[p] = len(self.backend[p])

    # Sort physical qubits by degree descending
    physical_sorted = sorted(range(N), key=lambda p: physical_degree[p], reverse=True)

    # Pre-assignment: map i-th highest degree logical to i-th highest degree physical
    pre_assign = {}
    for i, l in enumerate(logical_sorted):
        if i < len(physical_sorted):
            pre_assign[l] = physical_sorted[i]
        else:
            pre_assign[l] = l

    # Build N×N cost matrix
    cost = np.zeros((N, N), dtype=np.float64)

    for l in range(N):
        if l not in interaction_partners:
            continue
        for p in range(N):
            total_cost = 0.0
            for partner, weight in interaction_partners[l].items():
                partner_phys = pre_assign.get(partner, partner)
                if p < len(self.distance_matrix) and partner_phys < len(self.distance_matrix):
                    total_cost += weight * self.distance_matrix[p][partner_phys]
            cost[l][p] = total_cost

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)

    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    for l, p in zip(row_ind, col_ind):
        self.mapping_dict[l] = p

    # Build reverse mapping
    for l in range(N):
        self.reverse_mapping_dict[self.mapping_dict[l]] = l

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)