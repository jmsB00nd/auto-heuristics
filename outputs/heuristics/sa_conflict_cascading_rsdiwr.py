def init_mapping(self):
    import random
    import math
    from collections import defaultdict

    n = self.num_qubits

    # Identify logical qubits involved in 2-qubit gates
    if self.access2q is None:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    gates_2q = {g: qs for g, qs in self.access2q.items() if len(qs) == 2}

    if not gates_2q:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Build conflict graph: logical qubits that share a 2-qubit gate
    conflict_adj = defaultdict(set)
    qubit_gates = defaultdict(list)
    for g, qs in gates_2q.items():
        q1, q2 = qs[0], qs[1]
        conflict_adj[q1].add(q2)
        conflict_adj[q2].add(q1)
        qubit_gates[q1].append(g)
        qubit_gates[q2].append(g)

    logical_qubits = sorted(conflict_adj.keys())

    # Collect conflict-graph edges as candidate swap pairs (logical qubit pairs)
    conflict_edges = []
    seen = set()
    for q in logical_qubits:
        for nb in conflict_adj[q]:
            edge = (min(q, nb), max(q, nb))
            if edge not in seen:
                seen.add(edge)
                conflict_edges.append(edge)

    if not conflict_edges:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # RSDIWR congestion-cascading cost function
    dist = self.distance_matrix
    gate_list = list(gates_2q.items())

    def compute_cost(mapping):
        # Per-physical-qubit congestion: sum of distances for all gates touching that qubit
        congestion = defaultdict(float)
        gate_dists = {}
        for g, qs in gate_list:
            p1, p2 = mapping[qs[0]], mapping[qs[1]]
            d = dist[p1][p2]
            gate_dists[g] = d
            congestion[p1] += d
            congestion[p2] += d

        # Cascading cost: gate distance weighted by product of endpoint congestions
        total = 0.0
        for g, qs in gate_list:
            p1, p2 = mapping[qs[0]], mapping[qs[1]]
            c1 = congestion[p1] if congestion[p1] > 0 else 1.0
            c2 = congestion[p2] if congestion[p2] > 0 else 1.0
            total += gate_dists[g] * (c1 * c2)
        return total

    # Initialize with a random permutation
    mapping = list(range(n))
    perm = list(range(n))
    random.shuffle(perm)
    mapping = perm[:]

    current_cost = compute_cost(mapping)
    best_mapping = mapping[:]
    best_cost = current_cost

    # Calibrate temperature: evaluate 50 random neighbor costs, use sqrt(variance)
    sample_costs = []
    num_samples = min(50, len(conflict_edges) * 5)
    for _ in range(num_samples):
        lq1, lq2 = random.choice(conflict_edges)
        mapping[lq1], mapping[lq2] = mapping[lq2], mapping[lq1]
        c = compute_cost(mapping)
        sample_costs.append(c)
        mapping[lq1], mapping[lq2] = mapping[lq2], mapping[lq1]

    if len(sample_costs) > 1:
        mean_c = sum(sample_costs) / len(sample_costs)
        variance = sum((c - mean_c) ** 2 for c in sample_costs) / (len(sample_costs) - 1)
        T0 = math.sqrt(variance) if variance > 0 else 1.0
    else:
        T0 = max(1.0, current_cost * 0.1)

    # SA parameters
    num_logical = len(logical_qubits)
    max_iterations = max(2000, num_logical * num_logical * 10)
    cooling_rate = 0.995
    T = T0
    T_min = T0 * 1e-6

    for iteration in range(max_iterations):
        if T < T_min:
            break

        lq1, lq2 = random.choice(conflict_edges)

        # Swap in mapping
        mapping[lq1], mapping[lq2] = mapping[lq2], mapping[lq1]
        new_cost = compute_cost(mapping)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_mapping = mapping[:]
        else:
            # Revert
            mapping[lq1], mapping[lq2] = mapping[lq2], mapping[lq1]

        T *= cooling_rate

    # Install best mapping
    self.mapping_dict = best_mapping[:]
    self.reverse_mapping_dict = [0] * n
    for logical, physical in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[physical] = logical

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)