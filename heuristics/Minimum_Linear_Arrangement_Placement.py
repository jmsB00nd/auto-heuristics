def init_mapping(self):
    from collections import defaultdict, deque

    # --- 1. Extract logical qubits and build interaction graph ---
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Build logical interaction graph (adjacency + weights)
    interaction_adj = defaultdict(set)
    interaction_weight = defaultdict(float)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            interaction_adj[q1].add(q2)
            interaction_adj[q2].add(q1)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    # --- 2. Solve MLA via barycenter heuristic ---
    # Initialize with BFS from highest-degree logical qubit
    n_logical = len(logical_qubits)
    degrees = {q: len(interaction_adj.get(q, set())) for q in logical_qubits}
    start_node = max(logical_qubits, key=lambda q: degrees[q])

    visited = set()
    bfs_order = []
    queue = deque([start_node])
    visited.add(start_node)
    while queue:
        node = queue.popleft()
        bfs_order.append(node)
        neighbors = sorted(interaction_adj.get(node, set()), key=lambda x: -degrees.get(x, 0))
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    for q in logical_qubits:
        if q not in visited:
            bfs_order.append(q)

    position = {q: i for i, q in enumerate(bfs_order)}

    # Iterative barycenter refinement (weighted by interaction counts)
    for _ in range(50):
        barycenters = {}
        for q in logical_qubits:
            neighbors = interaction_adj.get(q, set())
            if neighbors:
                total_w = 0.0
                weighted_pos = 0.0
                for nb in neighbors:
                    key = (min(q, nb), max(q, nb))
                    w = interaction_weight.get(key, 1.0)
                    weighted_pos += position[nb] * w
                    total_w += w
                barycenters[q] = weighted_pos / total_w
            else:
                barycenters[q] = position[q]

        sorted_qubits = sorted(logical_qubits, key=lambda q: barycenters[q])
        new_position = {q: i for i, q in enumerate(sorted_qubits)}
        if new_position == position:
            break
        position = new_position

    mla_ordering = sorted(logical_qubits, key=lambda q: position[q])

    # --- 3. BFS linearization of the hardware graph ---
    hw_degrees = {p: len(self.backend[p]) for p in physical_qubits}
    hw_start = max(physical_qubits, key=lambda p: hw_degrees[p])

    hw_visited = set()
    hw_bfs_order = []
    hw_queue = deque([hw_start])
    hw_visited.add(hw_start)
    while hw_queue:
        node = hw_queue.popleft()
        hw_bfs_order.append(node)
        neighbors = sorted(self.backend[node], key=lambda x: -hw_degrees.get(x, 0))
        for nb in neighbors:
            if nb not in hw_visited:
                hw_visited.add(nb)
                hw_queue.append(nb)
    for p in physical_qubits:
        if p not in hw_visited:
            hw_bfs_order.append(p)

    # --- 4. Map k-th MLA logical qubit to k-th HW BFS physical qubit ---
    self.mapping_dict = list(range(self.num_qubits))
    self.reverse_mapping_dict = list(range(self.num_qubits))

    for k in range(min(n_logical, len(hw_bfs_order))):
        lq = mla_ordering[k]
        pq = hw_bfs_order[k]
        current_phys = self.mapping_dict[lq]
        displaced_lq = self.reverse_mapping_dict[pq]
        self.mapping_dict[lq] = pq
        self.mapping_dict[displaced_lq] = current_phys
        self.reverse_mapping_dict[pq] = lq
        self.reverse_mapping_dict[current_phys] = displaced_lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)