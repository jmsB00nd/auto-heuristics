def init_mapping(self):
    """
    Critical Chain Hardware Path Co-Routing (CCHPCR)

    Strategy:
      1. Build a sequential 2Q-gate dependency DAG from self.access.
      2. Find the critical path (longest path by gate count) via DP on a
         topological sort — this is the circuit's routing bottleneck.
      3. Extract unique logical qubits along the critical path in first-
         appearance order → the "critical chain".
      4. Find an approximate hardware diameter path via two-sweep BFS
         (start → farthest node u → farthest node v) → the "hw chain".
      5. Co-route: map critical_chain[i] → hw_chain[i] for i = 0..k-1.
         Because hw_chain nodes are adjacent by construction, gates on the
         critical path become directly executable with zero SWAP overhead.
      6. Place remaining logical qubits by interaction-weighted BFS
         expansion: greedily pick the unplaced qubit with the highest
         total interaction weight to already-placed qubits and seat it on
         the hardware neighbour that minimises weighted distance cost.
      7. Commit via in-place swap bijection to guarantee a strict 1-to-1
         mapping over all num_qubits indices.
    """
    from collections import defaultdict, deque

    # ------------------------------------------------------------------ #
    # Step 1: Gather logical qubits and build interaction neighbourhood   #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
    interaction_neighbors = defaultdict(dict)   # lq → {lq2: weight}

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            w = interaction_neighbors[q1].get(q2, 0) + 1
            interaction_neighbors[q1][q2] = w
            interaction_neighbors[q2][q1] = w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Trivial fallback when the circuit has no gates at all
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 2: Build sequential 2Q dependency DAG                          #
    #                                                                      #
    # For each qubit, track the last 2Q gate that used it.  Each new 2Q   #
    # gate that uses the same qubit inherits a dependency edge from that   #
    # last gate, serialising per-qubit usage correctly.                   #
    # ------------------------------------------------------------------ #
    two_qubit_gates = {g: q for g, q in self.access.items() if len(q) == 2}

    lq_to_phys = {}   # populated in steps 5 & 6

    if not two_qubit_gates:
        # No 2Q gates → skip critical-path phase; go straight to BFS
        # expansion anchored on the highest-degree logical qubit.
        weighted_degree = defaultdict(float)
        for lq, nbrs in interaction_neighbors.items():
            weighted_degree[lq] = sum(nbrs.values())
        anchor_lq = max(logical_qubits, key=lambda q: weighted_degree[q])
        anchor_phys = min(
            physical_qubits,
            key=lambda p: sum(
                self.distance_matrix[p][o]
                for o in physical_qubits
                if self.distance_matrix[p][o] != float('inf')
            )
        )
        lq_to_phys[anchor_lq] = anchor_phys
    else:
        # ---------------------------------------------------------------- #
        # Step 3: Longest-path DP on the 2Q sequential DAG                 #
        # ---------------------------------------------------------------- #
        sorted_2q = sorted(two_qubit_gates.keys())

        last_gate_on_qubit = {}        # qubit → most-recent 2Q gate ID
        dag_succ = defaultdict(set)    # gate → set of successor gates
        dag_pred = defaultdict(set)    # gate → set of predecessor gates

        for gate in sorted_2q:
            q1, q2 = two_qubit_gates[gate]
            for q in (q1, q2):
                if q in last_gate_on_qubit:
                    pred = last_gate_on_qubit[q]
                    dag_succ[pred].add(gate)
                    dag_pred[gate].add(pred)
            last_gate_on_qubit[q1] = gate
            last_gate_on_qubit[q2] = gate

        # Kahn's topological sort + DP for longest path
        in_deg  = {g: len(dag_pred[g]) for g in sorted_2q}
        dp_len  = {g: 1    for g in sorted_2q}
        dp_prev = {g: None for g in sorted_2q}
        topo_q  = deque(g for g in sorted_2q if in_deg[g] == 0)

        while topo_q:
            node = topo_q.popleft()
            for succ in dag_succ[node]:
                if dp_len[node] + 1 > dp_len[succ]:
                    dp_len[succ]  = dp_len[node] + 1
                    dp_prev[succ] = node
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    topo_q.append(succ)

        critical_end = max(sorted_2q, key=lambda g: dp_len[g])
        critical_path_gates = []
        node = critical_end
        while node is not None:
            critical_path_gates.append(node)
            node = dp_prev[node]
        critical_path_gates.reverse()

        # ---------------------------------------------------------------- #
        # Step 4: Extract unique logical qubits along the critical path     #
        # ---------------------------------------------------------------- #
        seen_lq = set()
        critical_chain_lq = []
        for gate in critical_path_gates:
            for q in two_qubit_gates[gate]:
                if q not in seen_lq:
                    seen_lq.add(q)
                    critical_chain_lq.append(q)

        # ---------------------------------------------------------------- #
        # Step 5: Hardware diameter chain via two-sweep BFS                 #
        # ---------------------------------------------------------------- #
        def bfs_farthest(src):
            dist = {src: 0}
            q = deque([src])
            farthest = src
            while q:
                node = q.popleft()
                for nb in self.backend[node]:
                    if nb not in dist:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > dist[farthest]:
                            farthest = nb
                        q.append(nb)
            return farthest, dist

        def bfs_path(src, dst):
            """BFS shortest path from src to dst; returns node list."""
            if src == dst:
                return [src]
            parent = {src: None}
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in self.backend[node]:
                    if nb not in parent:
                        parent[nb] = node
                        if nb == dst:
                            path, cur = [], dst
                            while cur is not None:
                                path.append(cur)
                                cur = parent[cur]
                            return path[::-1]
                        q.append(nb)
            return [src]   # fallback: unreachable

        u, _ = bfs_farthest(physical_qubits[0])
        v, _ = bfs_farthest(u)
        hw_chain = bfs_path(u, v)

        # Co-route: critical_chain_lq[i] → hw_chain[i]
        k = min(len(critical_chain_lq), len(hw_chain))
        for i in range(k):
            lq_to_phys[critical_chain_lq[i]] = hw_chain[i]

    # ------------------------------------------------------------------ #
    # Step 6: Remaining qubits — interaction-weighted BFS expansion       #
    # ------------------------------------------------------------------ #
    placed_phys = set(lq_to_phys.values())
    unplaced    = [lq for lq in logical_qubits if lq not in lq_to_phys]

    while unplaced:
        next_lq = max(
            unplaced,
            key=lambda lq: sum(
                interaction_neighbors[lq].get(p_lq, 0)
                for p_lq in lq_to_phys
            )
        )

        candidates = list({
            nb
            for phys in placed_phys
            for nb in self.backend[phys]
            if nb not in placed_phys
        })
        if not candidates:
            candidates = [p for p in physical_qubits if p not in placed_phys]
        if not candidates:
            break

        def placement_cost(phys_c, _lq=next_lq):
            total = 0.0
            for p_lq, p_phys in lq_to_phys.items():
                w = interaction_neighbors[_lq].get(p_lq, 0)
                if w > 0:
                    d = self.distance_matrix[phys_c][p_phys]
                    total += w * (d if d != float('inf') else 1e9)
            return total

        best_phys = min(candidates, key=placement_cost)
        lq_to_phys[next_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.remove(next_lq)

    # ------------------------------------------------------------------ #
    # Step 7: Build strict 1-to-1 bijection via in-place swap             #
    # ------------------------------------------------------------------ #
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq]                   = target_phys
        mapping_dict[displaced_lq]         = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)