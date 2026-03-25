def init_mapping(self):
    from collections import defaultdict, deque

    # ── 1. Collect logical qubits and 2-qubit gate interactions ──────────
    logical_qubits = sorted({q for qubits in self.access.values() for q in qubits})
    num_logical = len(logical_qubits)

    if num_logical == 0:
        self.mapping_dict = {}
        self.reverse_mapping_dict = {}
        return

    two_qubit_gates = {}  # gate_id -> (q1, q2)
    for gate_id in sorted(self.access.keys()):
        qubits = self.access[gate_id]
        if len(qubits) == 2:
            two_qubit_gates[gate_id] = (qubits[0], qubits[1])

    physical_nodes = sorted(self.backend.keys())

    # Fallback: if no 2-qubit gates, use trivial mapping
    if not two_qubit_gates:
        self.mapping_dict = {}
        self.reverse_mapping_dict = {}
        for i, lq in enumerate(logical_qubits):
            pq = physical_nodes[i] if i < len(physical_nodes) else i
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ── 2. Build DAG over 2-qubit gates (sequential qubit dependencies) ──
    gate_ids = sorted(two_qubit_gates.keys())
    last_gate_on_qubit = {}
    successors = defaultdict(set)
    predecessors = defaultdict(set)

    for gid in gate_ids:
        q1, q2 = two_qubit_gates[gid]
        for q in (q1, q2):
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                successors[pred].add(gid)
                predecessors[gid].add(pred)
            last_gate_on_qubit[q] = gid

    # ── 3. Longest path via topological sort + DP ────────────────────────
    in_degree = {gid: len(predecessors[gid]) for gid in gate_ids}
    dp = {gid: 1 for gid in gate_ids}
    parent = {gid: None for gid in gate_ids}

    queue = deque(gid for gid in gate_ids if in_degree[gid] == 0)
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            if dp[node] + 1 > dp[succ]:
                dp[succ] = dp[node] + 1
                parent[succ] = node
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # ── 4. Extract critical path and its ordered logical qubits ──────────
    end_gate = max(gate_ids, key=lambda g: dp[g])
    critical_path_gates = []
    current = end_gate
    while current is not None:
        critical_path_gates.append(current)
        current = parent[current]
    critical_path_gates.reverse()

    critical_path_qubits = []
    seen_cp = set()
    for gid in critical_path_gates:
        q1, q2 = two_qubit_gates[gid]
        for q in (q1, q2):
            if q not in seen_cp:
                critical_path_qubits.append(q)
                seen_cp.add(q)

    # ── 5. Find a simple path in the hardware graph of matching length ───
    path_len = len(critical_path_qubits)

    def find_hw_path(start, length, visited):
        """DFS to find a simple path of exactly `length` nodes."""
        if length == 1:
            return [start]
        visited.add(start)
        # Prefer neighbors with higher degree (less likely to dead-end)
        for nb in sorted(self.backend[start], key=lambda n: len(self.backend[n]), reverse=True):
            if nb not in visited:
                result = find_hw_path(nb, length - 1, visited)
                if result is not None:
                    return [start] + result
        visited.remove(start)
        return None

    best_hw_path = None

    # Try from degree-1 endpoints first (natural chain starts), then high-degree
    endpoints = [n for n in physical_nodes if len(self.backend[n]) == 1]
    high_degree = sorted(physical_nodes, key=lambda n: len(self.backend[n]), reverse=True)
    candidates_to_try = endpoints[:5] + high_degree[:5]
    seen_starts = set()

    for start in candidates_to_try:
        if start in seen_starts:
            continue
        seen_starts.add(start)
        result = find_hw_path(start, path_len, set())
        if result is not None:
            best_hw_path = result
            break

    # If exact length not found, find the longest path we can
    if best_hw_path is None:
        longest = []
        for start in candidates_to_try[:6]:
            visited = set()
            # Greedy longest walk: always go to unvisited neighbor with highest degree
            path = [start]
            visited.add(start)
            while True:
                nbs = [n for n in self.backend[path[-1]] if n not in visited]
                if not nbs:
                    break
                nxt = max(nbs, key=lambda n: len(self.backend[n]))
                path.append(nxt)
                visited.add(nxt)
            if len(path) > len(longest):
                longest = path
        best_hw_path = longest

    # ── 6. Map critical-path qubits to the hardware path ─────────────────
    used_physical = set()
    self.mapping_dict = {}
    self.reverse_mapping_dict = {}

    for i, lq in enumerate(critical_path_qubits):
        if i < len(best_hw_path):
            pq = best_hw_path[i]
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_physical.add(pq)

    # ── 7. Greedily assign remaining logical qubits ──────────────────────
    # Build interaction adjacency for greedy placement
    interactions = defaultdict(set)
    for gid, (q1, q2) in two_qubit_gates.items():
        interactions[q1].add(q2)
        interactions[q2].add(q1)

    remaining_logical = [q for q in logical_qubits if q not in self.mapping_dict]
    available_physical = set(physical_nodes) - used_physical

    # Sort remaining by number of already-placed partners (descending)
    # so qubits with more placed neighbors get assigned first
    def placement_priority(lq):
        return sum(1 for partner in interactions[lq] if partner in self.mapping_dict)

    remaining_logical.sort(key=placement_priority, reverse=True)

    for lq in remaining_logical:
        # Find placed interaction partners
        placed_partners = [self.mapping_dict[p] for p in interactions[lq] if p in self.mapping_dict]

        if placed_partners:
            best_pq = min(available_physical,
                          key=lambda p: sum(self.distance_matrix[p][pp] for pp in placed_partners))
        else:
            # No partners placed yet; pick closest to any used physical qubit
            if used_physical:
                best_pq = min(available_physical,
                              key=lambda p: min(self.distance_matrix[p][u] for u in used_physical))
            else:
                best_pq = min(available_physical)

        self.mapping_dict[lq] = best_pq
        self.reverse_mapping_dict[best_pq] = lq
        used_physical.add(best_pq)
        available_physical.remove(best_pq)

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)