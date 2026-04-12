def init_mapping(self):
    from collections import defaultdict, deque

    num_qubits = self.num_qubits

    # --- Step 1: Build 2-qubit gate DAG and identify logical qubits used ---
    # Extract only 2-qubit gates and build dependency graph
    two_q_gates = {}
    gate_qubits = {}
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            two_q_gates[gate_id] = qubits
            gate_qubits[gate_id] = qubits

    # Collect all logical qubits that appear in 2-qubit gates
    logical_qubits_used = set()
    for qubits in two_q_gates.values():
        logical_qubits_used.update(qubits)

    # If no 2-qubit gates, use trivial mapping
    if not two_q_gates:
        self.mapping_dict = list(range(num_qubits))
        self.reverse_mapping_dict = list(range(num_qubits))
        return

    # --- Step 2: Build DAG among 2-qubit gates using write-after-read/write dependencies ---
    # Track last gate that wrote to each qubit
    last_writer = {}
    # Track last gate that accessed each qubit
    last_accessor = {}
    
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    all_gate_ids = sorted(two_q_gates.keys())

    for gate_id in all_gate_ids:
        qubits = two_q_gates[gate_id]
        write_qubits = self.write_dict.get(gate_id, [])
        
        deps = set()
        for q in qubits:
            if q in last_writer:
                deps.add(last_writer[q])
            if q in last_accessor and last_accessor[q] != last_writer.get(q):
                deps.add(last_accessor[q])
        
        for dep in deps:
            if dep != gate_id:
                successors[dep].add(gate_id)
                predecessors[gate_id].add(dep)
        
        for q in qubits:
            last_accessor[q] = gate_id
        for q in write_qubits:
            if q in logical_qubits_used:
                last_writer[q] = gate_id

    # --- Step 3: Find critical path (longest path) via topological sort + DP ---
    # Topological sort
    in_degree = defaultdict(int)
    for g in all_gate_ids:
        if g not in in_degree:
            in_degree[g] = 0
    for g in all_gate_ids:
        for s in successors[g]:
            in_degree[s] += 1

    topo_order = []
    queue = deque([g for g in all_gate_ids if in_degree[g] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # DP for longest path
    dist = {g: 0 for g in all_gate_ids}
    parent = {g: None for g in all_gate_ids}

    for g in topo_order:
        for s in successors[g]:
            if dist[g] + 1 > dist[s]:
                dist[s] = dist[g] + 1
                parent[s] = g

    # Find the end of the critical path
    end_gate = max(all_gate_ids, key=lambda g: dist[g])

    # Trace back the critical path
    critical_path_gates = []
    g = end_gate
    while g is not None:
        critical_path_gates.append(g)
        g = parent[g]
    critical_path_gates.reverse()

    # --- Step 4: Extract ordered logical qubits along the critical path ---
    # Maintain order, deduplicate
    critical_logical_qubits = []
    seen = set()
    for gate_id in critical_path_gates:
        for q in two_q_gates[gate_id]:
            if q not in seen:
                critical_logical_qubits.append(q)
                seen.add(q)

    backbone_length = len(critical_logical_qubits)

    # --- Step 5: Find a physical path of at least backbone_length via BFS ---
    # Find the longest shortest-path in the hardware graph to anchor the backbone
    # Use BFS from each node to find paths of sufficient length
    best_path = None
    best_path_len = 0

    # Strategy: BFS to find a path of length >= backbone_length
    # Start from nodes with low degree (likely endpoints)
    physical_nodes = sorted(self.backend.keys())

    # Find a long path using double BFS (find diameter endpoints)
    def bfs_farthest(start):
        visited = {start: None}
        q = deque([start])
        farthest = start
        while q:
            node = q.popleft()
            farthest = node
            for nb in self.backend[node]:
                if nb not in visited:
                    visited[nb] = node
                    q.append(nb)
        # Reconstruct path
        path = []
        node = farthest
        while node is not None:
            path.append(node)
            node = visited[node]
        path.reverse()
        return path, farthest

    # Double BFS to find approximate diameter path
    _, far1 = bfs_farthest(physical_nodes[0])
    diameter_path, _ = bfs_farthest(far1)

    if len(diameter_path) >= backbone_length:
        best_path = diameter_path[:backbone_length]
    else:
        # If diameter path is shorter than needed, use what we have
        # and truncate backbone to fit
        best_path = diameter_path
        critical_logical_qubits = critical_logical_qubits[:len(best_path)]
        backbone_length = len(critical_logical_qubits)

    # --- Step 6: Anchor critical-path logical qubits onto physical backbone ---
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits
    placed_physical = set()
    placed_logical = set()

    for i in range(backbone_length):
        lq = critical_logical_qubits[i]
        pq = best_path[i]
        mapping[lq] = pq
        reverse_mapping[pq] = lq
        placed_physical.add(pq)
        placed_logical.add(lq)

    # --- Step 7: Place remaining logical qubits greedily with temporal-decay weighting ---
    # Build interaction graph with temporal decay
    # Count weighted interactions between logical qubits
    interaction_weight = defaultdict(float)
    total_gates = len(self.access)

    for idx, gate_id in enumerate(sorted(self.access.keys())):
        qubits = self.access[gate_id]
        if len(qubits) == 2:
            q1, q2 = qubits
            # Temporal decay: earlier gates matter more
            decay = 1.0 / (1.0 + idx * 0.01)
            interaction_weight[(q1, q2)] += decay
            interaction_weight[(q2, q1)] += decay

    # Remaining logical qubits that need placement (those involved in 2q gates first)
    remaining_logical = [q for q in sorted(logical_qubits_used) if q not in placed_logical]
    free_physical = set(physical_nodes) - placed_physical

    # Sort remaining by total interaction weight with already-placed qubits (descending)
    def placement_priority(lq):
        total = 0.0
        for plq in placed_logical:
            total += interaction_weight.get((lq, plq), 0.0)
        return -total  # negative for ascending sort = descending priority

    remaining_logical.sort(key=placement_priority)

    for lq in remaining_logical:
        # Score each free physical qubit by weighted proximity to placed neighbors
        best_pq = None
        best_score = float('inf')

        for pq in free_physical:
            score = 0.0
            for plq in placed_logical:
                w = interaction_weight.get((lq, plq), 0.0)
                if w > 0:
                    ppq = mapping[plq]
                    score += w * self.distance_matrix[pq][ppq]
            if score < best_score:
                best_score = score
                best_pq = pq

        if best_pq is None:
            # Just pick any free physical qubit
            best_pq = min(free_physical)

        mapping[lq] = best_pq
        reverse_mapping[best_pq] = lq
        placed_physical.add(best_pq)
        placed_logical.add(lq)
        free_physical.discard(best_pq)

    # --- Step 8: Pairwise swap refinement on placed qubits ---
    # Quick local search: try swapping pairs and keep if it reduces total weighted distance
    placed_list = [q for q in sorted(logical_qubits_used)]

    def compute_total_cost(m):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            if q1 < q2 and m[q1] != -1 and m[q2] != -1:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    current_cost = compute_total_cost(mapping)
    improved = True
    max_rounds = 3
    round_count = 0

    while improved and round_count < max_rounds:
        improved = False
        round_count += 1
        for i in range(len(placed_list)):
            for j in range(i + 1, len(placed_list)):
                lq1 = placed_list[i]
                lq2 = placed_list[j]
                pq1 = mapping[lq1]
                pq2 = mapping[lq2]

                # Try swap
                mapping[lq1], mapping[lq2] = pq2, pq1
                new_cost = compute_total_cost(mapping)

                if new_cost < current_cost:
                    # Keep swap, update reverse mapping
                    reverse_mapping[pq1] = lq2
                    reverse_mapping[pq2] = lq1
                    current_cost = new_cost
                    improved = True
                else:
                    # Revert
                    mapping[lq1], mapping[lq2] = pq1, pq2

    # --- Step 9: Fill any unmapped logical qubits (those not in any 2q gate) ---
    all_logical = set(range(num_qubits))
    unmapped_logical = sorted(all_logical - placed_logical)
    free_physical = sorted(set(physical_nodes) - placed_physical)

    # Also handle logical qubits beyond physical nodes range
    for i, lq in enumerate(unmapped_logical):
        if i < len(free_physical):
            pq = free_physical[i]
            mapping[lq] = pq
            reverse_mapping[pq] = lq
        else:
            # Should not happen if num_qubits is correct
            mapping[lq] = lq
            reverse_mapping[lq] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping