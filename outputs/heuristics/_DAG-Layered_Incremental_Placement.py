def init_mapping(self):
    from collections import defaultdict, deque

    n_physical = self.num_qubits
    physical_nodes = sorted(self.backend.keys())

    # --- Step 1: Identify logical qubits and 2-qubit gates ---
    logical_qubits_set = set()
    two_qubit_gates = {}
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubits_set.add(q)
        if len(qubits) == 2:
            two_qubit_gates[gate] = (qubits[0], qubits[1])

    # --- Step 2: Build a simple DAG from gate ordering per qubit ---
    all_gates = set(self.access.keys())
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    last_gate_per_qubit = {}

    for gate in sorted(all_gates):
        for q in self.access[gate]:
            if q in last_gate_per_qubit:
                prev = last_gate_per_qubit[q]
                if prev != gate:
                    successors[prev].add(gate)
                    predecessors[gate].add(prev)
            last_gate_per_qubit[q] = gate

    # --- Step 3: Topological layering via BFS ---
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    layers = []
    current_layer = sorted([g for g in all_gates if in_degree[g] == 0])

    while current_layer:
        layers.append(current_layer)
        next_layer_set = set()
        for g in current_layer:
            for s in successors.get(g, set()):
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    next_layer_set.add(s)
        current_layer = sorted(next_layer_set)

    # --- Step 4: Incremental placement processing layers front-to-back ---
    placed = {}       # logical -> physical
    occupied = set()  # set of used physical qubits

    def closest_unoccupied_bfs(target_phys):
        """BFS from target_phys to find nearest unoccupied physical qubit."""
        if target_phys not in occupied:
            return target_phys
        visited = {target_phys}
        queue = deque([target_phys])
        while queue:
            node = queue.popleft()
            for neighbor in self.backend.get(node, []):
                if neighbor not in visited:
                    if neighbor not in occupied:
                        return neighbor
                    visited.add(neighbor)
                    queue.append(neighbor)
        for p in physical_nodes:
            if p not in occupied:
                return p
        return None

    def best_unoccupied_edge():
        """Find highest-degree unoccupied edge in the hardware graph."""
        best_edge = None
        best_score = -1
        for (u, v) in self.backend_connections:
            if u not in occupied and v not in occupied:
                score = len(self.backend.get(u, [])) + len(self.backend.get(v, []))
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
        return best_edge

    for layer in layers:
        layer_2q = [g for g in layer if g in two_qubit_gates]

        one_placed = []   # (new_qubit, placed_partner)
        both_new = []     # (q1, q2)

        for g in layer_2q:
            q1, q2 = two_qubit_gates[g]
            q1_new = q1 not in placed
            q2_new = q2 not in placed

            if not q1_new and not q2_new:
                continue
            elif q1_new and not q2_new:
                one_placed.append((q1, q2))
            elif q2_new and not q1_new:
                one_placed.append((q2, q1))
            else:
                both_new.append((q1, q2))

        # Place qubits with an already-placed partner first
        for new_q, partner in one_placed:
            if new_q in placed:
                continue
            phys = closest_unoccupied_bfs(placed[partner])
            if phys is not None:
                placed[new_q] = phys
                occupied.add(phys)

        # Place pairs where both qubits are new
        for q1, q2 in both_new:
            q1_done = q1 in placed
            q2_done = q2 in placed

            if q1_done and q2_done:
                continue
            if q1_done and not q2_done:
                phys = closest_unoccupied_bfs(placed[q1])
                if phys is not None:
                    placed[q2] = phys
                    occupied.add(phys)
                continue
            if q2_done and not q1_done:
                phys = closest_unoccupied_bfs(placed[q2])
                if phys is not None:
                    placed[q1] = phys
                    occupied.add(phys)
                continue

            # Both truly new: place on highest-degree unoccupied edge
            edge = best_unoccupied_edge()
            if edge is not None:
                placed[q1] = edge[0]
                placed[q2] = edge[1]
                occupied.add(edge[0])
                occupied.add(edge[1])
            else:
                for q in (q1, q2):
                    if q not in placed:
                        for p in physical_nodes:
                            if p not in occupied:
                                placed[q] = p
                                occupied.add(p)
                                break

    # --- Step 5: Build full mapping lists ---
    mapping_dict_list = list(range(n_physical))
    reverse_mapping_dict_list = list(range(n_physical))
    assigned_logical = set()

    for lq, pq in placed.items():
        mapping_dict_list[lq] = pq
        reverse_mapping_dict_list[pq] = lq
        assigned_logical.add(lq)

    remaining_physical = [p for p in range(n_physical) if p not in occupied]
    remaining_logical = [q for q in range(n_physical) if q not in assigned_logical]
    for lq, pq in zip(remaining_logical, remaining_physical):
        mapping_dict_list[lq] = pq
        reverse_mapping_dict_list[pq] = lq

    # --- Step 6: Populate mapping ---
    self.mapping_dict = mapping_dict_list
    self.reverse_mapping_dict = reverse_mapping_dict_list

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)