def init_mapping(self):
    from collections import defaultdict, deque

    two_q_gates = {}
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            two_q_gates[gate_id] = (qubits[0], qubits[1])

    if not two_q_gates:
        self.mapping_dict, self.reverse_mapping_dict = generate_trivial_initial_mapping(self.num_qubits)
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    qubit_gate_chains = defaultdict(list)
    for gate_id in sorted(two_q_gates.keys()):
        for q in two_q_gates[gate_id]:
            qubit_gate_chains[q].append(gate_id)

    successors = defaultdict(set)
    in_degree = {g: 0 for g in two_q_gates}

    for q, chain in qubit_gate_chains.items():
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            if b not in successors[a]:
                successors[a].add(b)
                in_degree[b] += 1

    layers = []
    queue = deque(g for g in two_q_gates if in_degree[g] == 0)
    while queue:
        layer = list(queue)
        queue = deque()
        layer.sort(
            key=lambda g: -self.qubit_interaction_graph[two_q_gates[g][0]].get(
                two_q_gates[g][1], 0
            )
        )
        layers.append(layer)
        for g in layer:
            for s in successors[g]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)

    n = self.num_qubits
    dm = self.distance_matrix
    dm_size = len(dm)

    def dist(p1, p2):
        if p1 < dm_size and p2 < dm_size:
            return dm[p1][p2]
        return float('inf')

    placed_logical = set()
    used_physical = set()
    mapping = [-1] * n

    def place(logical_q, physical_q):
        mapping[logical_q] = physical_q
        placed_logical.add(logical_q)
        used_physical.add(physical_q)

    def neighbor_cost(logical_q, physical_q):
        cost = 0.0
        for nbr_q, weight in self.qubit_interaction_graph[logical_q].items():
            if nbr_q in placed_logical:
                cost += weight * dist(physical_q, mapping[nbr_q])
        return cost

    def best_free_near(anchor_physical, exclude=None):
        best_p, best_d = None, float('inf')
        for p in range(n):
            if p in used_physical or p == exclude:
                continue
            d = dist(anchor_physical, p)
            if d < best_d:
                best_d = d
                best_p = p
        return best_p

    def most_central_free(exclude=None):
        best_p, best_c = None, -1.0
        for p in range(n):
            if p in used_physical or p == exclude:
                continue
            c = self.physical_centrality.get(p, 0)
            if c > best_c:
                best_c = c
                best_p = p
        return best_p

    for layer in layers:
        for gate_id in layer:
            q1, q2 = two_q_gates[gate_id]
            q1_placed = q1 in placed_logical
            q2_placed = q2 in placed_logical

            if q1_placed and q2_placed:
                continue

            if not q1_placed and not q2_placed:
                best_edge = None
                best_score = float('inf')
                for p1 in range(n):
                    if p1 in used_physical:
                        continue
                    for p2 in self.backend.get(p1, set()):
                        if p2 in used_physical:
                            continue
                        score = (
                            neighbor_cost(q1, p1)
                            + neighbor_cost(q2, p2)
                            - 0.001
                            * (
                                self.physical_centrality.get(p1, 0)
                                + self.physical_centrality.get(p2, 0)
                            )
                        )
                        if score < best_score:
                            best_score = score
                            best_edge = (p1, p2)

                if best_edge is not None:
                    place(q1, best_edge[0])
                    place(q2, best_edge[1])
                else:
                    p1 = most_central_free()
                    if p1 is not None:
                        place(q1, p1)
                    p2 = (
                        best_free_near(mapping[q1])
                        if q1 in placed_logical
                        else most_central_free()
                    )
                    if p2 is not None:
                        place(q2, p2)

            elif q1_placed:
                p = best_free_near(mapping[q1])
                if p is not None:
                    place(q2, p)

            else:
                p = best_free_near(mapping[q2])
                if p is not None:
                    place(q1, p)

    remaining_physical = sorted(set(range(n)) - used_physical)
    idx = 0
    for q in range(n):
        if q not in placed_logical:
            mapping[q] = remaining_physical[idx]
            idx += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = [0] * n
    for logical_q in range(n):
        self.reverse_mapping_dict[mapping[logical_q]] = logical_q

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)