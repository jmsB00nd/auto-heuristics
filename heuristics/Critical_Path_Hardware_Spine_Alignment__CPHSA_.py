def init_mapping(self):
    """
    Critical Path Hardware Spine Alignment (CPHSA)

    1. Build a per-qubit sequential DAG from self.access (gate order = sorted gate IDs).
    2. Find the critical path (longest dependency chain) via topological DP.
    3. Extract the ordered sequence of unique logical qubits from 2-qubit gates on that path.
    4. Find the hardware "spine" = diameter path of the backend topology
       (longest shortest-path chain), using the precomputed distance_matrix.
    5. Align critical-path qubits 1-to-1 onto the spine.
    6. Place remaining qubits greedily via BFS wave: pick the unplaced logical qubit
       with the most interaction weight to already-placed qubits, assign it to the
       hardware neighbor of placed qubits that minimises weighted distance cost.
    7. Build a strict bijection (identity + swap-in-place) over all num_qubits indices.
    """
    from collections import defaultdict, deque

    # ── Collect logical qubits and build interaction graph ──────────────────
    logical_qubit_set = set()
    interaction_neighbors = defaultdict(dict)   # lq -> {lq: weight}
    gate_order_per_qubit = defaultdict(list)    # qubit -> [gate_id, ...] sorted

    for gate, qubits in sorted(self.access.items()):   # sorted = program order
        for q in qubits:
            logical_qubit_set.add(q)
            gate_order_per_qubit[q].append(gate)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0) + 1
            interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0) + 1

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback: no gates → identity mapping
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ── Step 1: Build sequential DAG (per-qubit gate chains) ────────────────
    # edge: gate_i → gate_{i+1} for each qubit's ordered gate list
    all_gates = set(self.access.keys())
    dag_succ = defaultdict(set)
    dag_pred = defaultdict(set)
    for gate in all_gates:          # ensure every gate has an entry
        dag_succ[gate]
        dag_pred[gate]
    for gates in gate_order_per_qubit.values():
        for i in range(len(gates) - 1):
            src, dst = gates[i], gates[i + 1]
            dag_succ[src].add(dst)
            dag_pred[dst].add(src)

    # ── Step 2: Topological sort + DP for longest path ──────────────────────
    in_deg = {g: len(dag_pred[g]) for g in all_gates}
    queue = deque(g for g in all_gates if in_deg[g] == 0)
    topo_order = []
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in dag_succ[g]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    longest_from = {g: 0 for g in all_gates}    # longest path length from g
    next_on_path = {g: None for g in all_gates}  # next gate on critical path
    for g in reversed(topo_order):
        best_len, best_next = 0, None
        for s in dag_succ[g]:
            candidate = 1 + longest_from[s]
            if candidate > best_len:
                best_len, best_next = candidate, s
        longest_from[g] = best_len
        next_on_path[g] = best_next

    # ── Step 3: Trace the critical path ─────────────────────────────────────
    start_gate = max(all_gates, key=lambda g: longest_from[g])
    critical_path = []
    cur = start_gate
    while cur is not None:
        critical_path.append(cur)
        cur = next_on_path[cur]

    # ── Step 4: Extract ordered unique qubits from 2-qubit gates on path ────
    cp_qubits_ordered = []
    cp_qubits_set = set()
    for gate in critical_path:
        if len(self.access[gate]) == 2:
            for q in self.access[gate]:
                if q not in cp_qubits_set:
                    cp_qubits_ordered.append(q)
                    cp_qubits_set.add(q)

    # Fallback: all qubits touched by critical path (covers 1-qubit-only paths)
    if not cp_qubits_ordered:
        for gate in critical_path:
            for q in self.access[gate]:
                if q not in cp_qubits_set:
                    cp_qubits_ordered.append(q)
                    cp_qubits_set.add(q)

    # ── Step 5: Find hardware spine = diameter path ──────────────────────────
    # Diameter endpoints: the pair of physical qubits with maximum distance
    diameter, spine_src, spine_dst = -1, physical_qubits[0], physical_qubits[0]
    for i, p1 in enumerate(physical_qubits):
        for p2 in physical_qubits[i + 1:]:
            d = self.distance_matrix[p1][p2]
            if d != float('inf') and d > diameter:
                diameter, spine_src, spine_dst = d, p1, p2

    # BFS to recover the actual spine path from src to dst
    def bfs_path(src, dst):
        if src == dst:
            return [src]
        parent = {src: None}
        bfs_q = deque([src])
        found = False
        while bfs_q and not found:
            cur = bfs_q.popleft()
            for nb in self.backend[cur]:
                if nb not in parent:
                    parent[nb] = cur
                    if nb == dst:
                        found = True
                        break
                    bfs_q.append(nb)
        path, node = [], dst
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()
        return path

    spine = bfs_path(spine_src, spine_dst)

    # ── Step 6: Align critical-path qubits onto the spine ───────────────────
    lq_to_phys = {}
    placed_phys = set()
    for i, lq in enumerate(cp_qubits_ordered):
        if i >= len(spine):
            break
        lq_to_phys[lq] = spine[i]
        placed_phys.add(spine[i])

    # ── Step 7: BFS-wave greedy placement for remaining qubits ───────────────
    unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

    while unplaced:
        # Pick logical qubit with most interaction to already-placed qubits
        next_lq = max(
            unplaced,
            key=lambda lq: sum(interaction_neighbors[lq].get(pl, 0)
                               for pl in lq_to_phys)
        )

        # Prefer hardware neighbors of placed qubits (BFS frontier)
        candidates = [
            nb
            for phys in placed_phys
            for nb in self.backend[phys]
            if nb not in placed_phys
        ]
        if not candidates:
            candidates = [p for p in physical_qubits if p not in placed_phys]
        if not candidates:
            break

        # Score: sum of w * dist(candidate, placed partner) for interacting partners
        def placement_cost(phys_c, lq=next_lq):
            cost = 0.0
            for pl_lq, pl_phys in lq_to_phys.items():
                w = interaction_neighbors[lq].get(pl_lq, 0)
                if w > 0:
                    d = self.distance_matrix[phys_c][pl_phys]
                    cost += w * (d if d != float('inf') else 1e9)
            return cost

        best_phys = min(candidates, key=placement_cost)
        lq_to_phys[next_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.remove(next_lq)

    # ── Step 8: Build strict 1-to-1 bijection over all num_qubits indices ────
    # Start from identity then swap-in-place for each CPHSA assignment.
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)