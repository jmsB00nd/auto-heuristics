def init_mapping(self):
    import networkx as nx
    from collections import defaultdict, deque

    n = self.num_qubits

    # Default identity
    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))

    if self.access2q is None or len(self.access2q) == 0:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Collect logical qubits that appear in 2q gates
    logical_qubits_in_2q = set()
    for gate, qubits in self.access2q.items():
        if len(qubits) == 2:
            logical_qubits_in_2q.update(qubits)

    if not logical_qubits_in_2q:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Build DAG of 2q gates: gate -> successor gates (based on shared qubits and order)
    # Group 2q gates by qubit
    qubit_gates = defaultdict(list)
    gate_order = list(self.access2q.keys())
    gate_index = {g: i for i, g in enumerate(gate_order)}

    for gate in gate_order:
        qubits = self.access2q[gate]
        if len(qubits) == 2:
            for q in qubits:
                qubit_gates[q].append(gate)

    successors = defaultdict(set)
    predecessors = defaultdict(set)
    all_2q_gates = [g for g in gate_order if len(self.access2q[g]) == 2]

    for q, gates in qubit_gates.items():
        for i in range(len(gates) - 1):
            g1, g2 = gates[i], gates[i + 1]
            if g2 not in successors[g1]:
                successors[g1].add(g2)
                predecessors[g2].add(g1)

    # Compute longest path (critical path) via topological order
    in_degree = defaultdict(int)
    for g in all_2q_gates:
        in_degree[g] = len(predecessors[g])

    topo_order = []
    queue = deque([g for g in all_2q_gates if in_degree[g] == 0])
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in successors[g]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    dist = {g: 0 for g in all_2q_gates}
    parent = {g: None for g in all_2q_gates}
    for g in topo_order:
        for s in successors[g]:
            if dist[g] + 1 > dist[s]:
                dist[s] = dist[g] + 1
                parent[s] = g

    # Trace back critical path
    if not dist:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    end_gate = max(dist, key=dist.get)
    critical_path_gates = []
    g = end_gate
    while g is not None:
        critical_path_gates.append(g)
        g = parent[g]
    critical_path_gates.reverse()

    # Extract ordered logical qubits along critical path (deduplicated, order-preserving)
    cp_logical = []
    seen_logical = set()
    for gate in critical_path_gates:
        for q in self.access2q[gate]:
            if q not in seen_logical:
                cp_logical.append(q)
                seen_logical.add(q)

    # Build hardware graph with networkx for diameter path
    G = nx.Graph()
    for u, v in self.backend_connections:
        G.add_edge(u, v)

    # Find diameter path: longest shortest path
    # Use BFS from each node to find eccentricity
    best_path = []
    best_len = -1
    nodes = list(G.nodes())

    for src in nodes:
        lengths = nx.single_source_shortest_path_length(G, src)
        farthest = max(lengths, key=lengths.get)
        if lengths[farthest] > best_len:
            best_len = lengths[farthest]
            best_src = src
            best_dst = farthest

    diameter_path = nx.shortest_path(G, best_src, best_dst)

    # Place critical-path qubits along the diameter path, centered
    used_physical = set()
    num_cp = len(cp_logical)

    if num_cp <= len(diameter_path):
        start_offset = (len(diameter_path) - num_cp) // 2
        for i, lq in enumerate(cp_logical):
            pq = diameter_path[start_offset + i]
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)
    else:
        # More CP qubits than diameter path length: place what fits, rest go to greedy
        for i, pq in enumerate(diameter_path):
            lq = cp_logical[i]
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)
        # Remaining CP qubits treated as unplaced
        seen_logical = set(cp_logical[:len(diameter_path)])
        cp_logical = cp_logical[:len(diameter_path)]

    placed_logical = set(cp_logical) if num_cp <= len(diameter_path) else set(cp_logical[:len(diameter_path)])

    # Build interaction weights between logical qubits
    interaction_weight = defaultdict(lambda: defaultdict(float))
    for gate in all_2q_gates:
        q1, q2 = self.access2q[gate]
        interaction_weight[q1][q2] += 1.0
        interaction_weight[q2][q1] += 1.0

    # Greedy placement of remaining logical qubits
    unplaced = [q for q in logical_qubits_in_2q if q not in placed_logical]

    while unplaced:
        best_qubit = None
        best_weight = -1.0
        for q in unplaced:
            w = sum(interaction_weight[q][p] for p in placed_logical if p in interaction_weight[q])
            if w > best_weight:
                best_weight = w
                best_qubit = q

        if best_qubit is None:
            break

        # Find nearest available physical qubit by weighted distance to placed partners
        best_pq = None
        best_cost = float('inf')
        available = [p for p in range(n) if p not in used_physical and p in G.nodes()]

        if not available:
            break

        for pq in available:
            cost = 0.0
            for partner in placed_logical:
                if partner in interaction_weight[best_qubit]:
                    w = interaction_weight[best_qubit][partner]
                    partner_pq = self.mapping_dict[partner]
                    if partner_pq < len(self.distance_matrix) and pq < len(self.distance_matrix):
                        cost += w * self.distance_matrix[pq][partner_pq]
            if cost < best_cost:
                best_cost = cost
                best_pq = pq

        if best_pq is not None:
            self.mapping_dict[best_qubit] = best_pq
            self.reverse_mapping_dict[best_pq] = best_qubit
            used_physical.add(best_pq)
            placed_logical.add(best_qubit)

        unplaced.remove(best_qubit)

    # Fill remaining unmapped logical qubits with unused physical qubits (identity fallback)
    all_logical = set(range(n))
    remaining_logical = sorted(all_logical - placed_logical)
    remaining_physical = sorted(set(range(n)) - used_physical)

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)