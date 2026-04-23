def init_mapping(self):
    from collections import deque, defaultdict

    num_q = self.num_qubits
    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    logical_qubits_in_circuit = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            for q in qubits:
                logical_qubits_in_circuit.add(q)

    if not logical_qubits_in_circuit:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Find most-active logical qubit
    best_logical = max(logical_qubits_in_circuit, key=lambda q: self.logical_activity.get(q, 0))

    # Find most-central physical qubit
    best_physical = max(range(num_q), key=lambda p: self.physical_centrality.get(p, 0))

    # BFS on logical interaction graph
    logical_layers = []
    logical_visited = set()
    logical_queue = deque()
    logical_queue.append(best_logical)
    logical_visited.add(best_logical)
    while logical_queue:
        layer = []
        for _ in range(len(logical_queue)):
            node = logical_queue.popleft()
            layer.append(node)
            for neighbor in self.qubit_interaction_graph.get(node, {}):
                if neighbor not in logical_visited and neighbor in logical_qubits_in_circuit:
                    logical_visited.add(neighbor)
                    logical_queue.append(neighbor)
        logical_layers.append(layer)

    # BFS on physical backend graph
    physical_layers = []
    physical_visited = set()
    physical_queue = deque()
    physical_queue.append(best_physical)
    physical_visited.add(best_physical)
    while physical_queue:
        layer = []
        for _ in range(len(physical_queue)):
            node = physical_queue.popleft()
            layer.append(node)
            for neighbor in self.backend.get(node, []):
                if neighbor not in physical_visited:
                    physical_visited.add(neighbor)
                    physical_queue.append(neighbor)
        physical_layers.append(layer)

    mapped_logical = set()
    used_physical = set()

    max_depth = max(len(logical_layers), len(physical_layers))

    leftover_logical = []
    leftover_physical = []

    for depth in range(max_depth):
        l_layer = logical_layers[depth] if depth < len(logical_layers) else []
        p_layer = physical_layers[depth] if depth < len(physical_layers) else []

        # Add any leftover from previous layers
        l_candidates = leftover_logical + [q for q in l_layer if q not in mapped_logical]
        p_candidates = leftover_physical + [p for p in p_layer if p not in used_physical]

        # Sort logical qubits by descending interaction weight to already-placed qubits
        def logical_weight_to_placed(q):
            w = 0
            for neighbor, count in self.qubit_interaction_graph.get(q, {}).items():
                if neighbor in mapped_logical:
                    w += count
            return w

        l_candidates.sort(key=logical_weight_to_placed, reverse=True)

        # Sort physical qubits by descending adjacency count to already-assigned physical qubits
        def physical_adj_to_assigned(p):
            count = 0
            for neighbor in self.backend.get(p, []):
                if neighbor in used_physical:
                    count += 1
            return count

        p_candidates.sort(key=physical_adj_to_assigned, reverse=True)

        pairs = min(len(l_candidates), len(p_candidates))
        for i in range(pairs):
            lq = l_candidates[i]
            pq = p_candidates[i]
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            mapped_logical.add(lq)
            used_physical.add(pq)

        leftover_logical = l_candidates[pairs:]
        leftover_physical = p_candidates[pairs:]

    # Handle any logical qubits not reached by BFS (disconnected components)
    unreached = [q for q in logical_qubits_in_circuit if q not in mapped_logical]
    remaining_physical = [p for p in range(num_q) if p not in used_physical]

    # Also add any leftover from the layer loop
    all_unmapped_logical = leftover_logical + unreached
    all_remaining_physical = leftover_physical + remaining_physical

    for lq, pq in zip(all_unmapped_logical, all_remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        mapped_logical.add(lq)
        used_physical.add(pq)

    # Fallback: assign any still-unmapped logical qubits to remaining physical qubits
    still_unmapped = [q for q in range(num_q) if q not in mapped_logical]
    still_free = [p for p in range(num_q) if p not in used_physical]
    for lq, pq in zip(still_unmapped, still_free):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)