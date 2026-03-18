def qlosure_poly_heuristic(self, swap_gate):
    from collections import deque

    epsilon = 1.0
    W = 0.5

    # -----------------------------------------------------------------------
    # PERT slack precomputation — O(V+E), cached per DAG instance.
    # Invalidated automatically when dag2q is reassigned between iterations.
    # -----------------------------------------------------------------------
    dag_id = id(self.dag2q)
    if not hasattr(self, '_cpsic_dag_id') or self._cpsic_dag_id != dag_id:

        all_gates = set(self.dag2q.keys()) | set(self.dag_predecessors2q.keys())

        # --- Forward pass: longest path TO each gate (from sources) ---
        forward_depth = {g: 0 for g in all_gates}
        in_deg = {g: len(self.dag_predecessors2q.get(g, set()) & all_gates)
                  for g in all_gates}
        queue = deque(g for g in all_gates if in_deg[g] == 0)
        topo_order = []
        while queue:
            gate = queue.popleft()
            topo_order.append(gate)
            for succ in self.dag2q.get(gate, set()):
                if succ not in all_gates:
                    continue
                new_d = forward_depth[gate] + 1
                if new_d > forward_depth[succ]:
                    forward_depth[succ] = new_d
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        # --- Backward pass: longest path FROM each gate (to sinks) ---
        backward_depth = {g: 0 for g in all_gates}
        for gate in reversed(topo_order):
            for succ in self.dag2q.get(gate, set()):
                if succ not in all_gates:
                    continue
                new_d = backward_depth[succ] + 1
                if new_d > backward_depth[gate]:
                    backward_depth[gate] = new_d

        # total_span = length of the critical path (longest path in DAG)
        total_span = max(
            (forward_depth[g] + backward_depth[g] for g in all_gates),
            default=0
        )

        # slack(g) = scheduling latitude; 0 means on the critical path
        self._cpsic_slack = {
            g: max(0, total_span - forward_depth[g] - backward_depth[g])
            for g in all_gates
        }
        self._cpsic_dag_id = dag_id

    slack_cache = self._cpsic_slack
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # -----------------------------------------------------------------------
    # Front layer: w(g) = 1 / (slack(g) + ε)
    # Normalise by total weight W_F to get a pure distance signal.
    # -----------------------------------------------------------------------
    W_F = 0.0
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        w = 1.0 / (slack_cache.get(g, 0) + epsilon)
        W_F += w
        f_distance += w * self.distance_matrix[Q1][Q2]

    # -----------------------------------------------------------------------
    # Extended layer: further discounted by look-ahead layer index.
    # -----------------------------------------------------------------------
    W_E = 0.0
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        w = 1.0 / ((slack_cache.get(g, 0) + epsilon) * layer_factor)
        W_E += w
        e_distance += w * self.distance_matrix[Q1][Q2]

    f_norm = f_distance / W_F if W_F > 0.0 else 0.0
    e_norm = e_distance / W_E if W_E > 0.0 else 0.0

    H = max_decay * (f_norm + W * e_norm)

    # Deterministic tie-breaking — prevents swap oscillation with negligible cost distortion
    H += (swap_gate[0] * self.num_qubits + swap_gate[1]) * 1e-12

    return H