def qlosure_poly_heuristic(self, swap_gate):
    from collections import deque

    # ── Cache ASAP/ALAP: recompute only when front_layer changes ──────────
    # Within one routing step, many candidate swaps are evaluated but the
    # DAG topology is frozen – only temp_mapping_dict varies.
    cache_key = frozenset(self.front_layer)
    if not hasattr(self, '_ssuc_key') or self._ssuc_key != cache_key:

        # Step 1 – BFS to collect every 2q gate reachable from front_layer
        all_gates = set()
        stack = list(self.front_layer)
        while stack:
            g = stack.pop()
            if g in all_gates:
                continue
            all_gates.add(g)
            for s in self.dag2q.get(g, set()):
                stack.append(s)

        # Step 2 – ASAP via forward topological relaxation
        #   in_degree counts only predecessors that are still pending
        #   (i.e., also reachable from the current front_layer).
        in_degree = {
            g: sum(1 for p in self.dag_predecessors2q.get(g, set()) if p in all_gates)
            for g in all_gates
        }
        asap = {}
        topo_q = deque()
        for g in all_gates:
            if in_degree[g] == 0:          # no pending 2q predecessor → ready now
                asap[g] = 0
                topo_q.append(g)

        while topo_q:
            g = topo_q.popleft()
            for s in self.dag2q.get(g, set()):
                if s not in all_gates:
                    continue
                asap[s] = max(asap.get(s, 0), asap[g] + 1)
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    topo_q.append(s)

        # Step 3 – ALAP via backward topological relaxation
        #   depth_to_end[g] = length of longest path from g to any sink
        #   ALAP[g] = total_remaining_depth - depth_to_end[g]
        out_degree = {
            g: sum(1 for s in self.dag2q.get(g, set()) if s in all_gates)
            for g in all_gates
        }
        depth_to_end = {}
        back_q = deque()
        for g in all_gates:
            if out_degree[g] == 0:         # sink: no pending 2q successor
                depth_to_end[g] = 0
                back_q.append(g)

        total_depth = max(asap.values(), default=0)

        while back_q:
            g = back_q.popleft()
            for p in self.dag_predecessors2q.get(g, set()):
                if p not in all_gates:
                    continue
                depth_to_end[p] = max(depth_to_end.get(p, 0), depth_to_end[g] + 1)
                out_degree[p] -= 1
                if out_degree[p] == 0:
                    back_q.append(p)

        alap = {g: total_depth - depth_to_end.get(g, 0) for g in all_gates}

        # Persist cache
        self._ssuc_key   = cache_key
        self._ssuc_asap  = asap
        self._ssuc_alap  = alap

    asap = self._ssuc_asap
    alap = self._ssuc_alap

    # ── Cost accumulation ─────────────────────────────────────────────────
    front_layer_size   = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    max_decay = max(self.decay_parameter[swap_gate[0]],
                    self.decay_parameter[swap_gate[1]])

    # Front layer: weight each gate's distance by its scheduling urgency.
    # Gates on the critical path (slack = 0) have urgency = 1.0 (maximum).
    # Gates with slack k have urgency = 1/(k+1), shrinking fast.
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        slack    = max(0, alap.get(g, 0) - asap.get(g, 0))
        urgency  = 1.0 / (slack + 1.0)
        f_distance += urgency * self.distance_matrix[Q1][Q2]

    # Extended lookahead layer: same urgency weighting, decayed by depth.
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        slack        = max(0, alap.get(g, 0) - asap.get(g, 0))
        urgency      = 1.0 / (slack + 1.0)
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        e_distance  += urgency * self.distance_matrix[Q1][Q2] / layer_factor

    W = 1.0
    H = max_decay * (
        f_distance / front_layer_size
        + W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )
    return H