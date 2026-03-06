def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Compute immediate successor fan-out for each gate (from forward DAG)
    all_gates = list(self.front_layer) + list(self.extended_layer)
    fanout = {g: len(self.dag2q.get(g, [])) for g in all_gates}

    max_fo = max(fanout.values(), default=1)
    max_fo = max(max_fo, 1)  # guard against all-zero case

    # Front layer: distance weighted by normalized fan-out urgency
    # High fan-out => high urgency => distance penalized more => prefer routing it sooner
    f_cost = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        norm_fo = fanout[g] / max_fo           # in [0, 1]
        urgency = 1.0 + norm_fo                # in [1, 2]
        f_cost += urgency * dist

    # Extended layer: same urgency weighting, attenuated by depth
    e_cost = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        norm_fo = fanout[g] / max_fo
        urgency = 1.0 + norm_fo
        e_cost += urgency * dist / layer_factor

    # Dynamic W: if extended layer has relatively higher fan-out than front layer,
    # give it more lookahead weight (those gates unlock more downstream work)
    avg_front_fo = (sum(fanout[g] for g in self.front_layer) / front_layer_size
                    if front_layer_size else 0)
    avg_ext_fo   = (sum(fanout[g] for g in self.extended_layer) / extended_layer_size
                    if extended_layer_size else 0)
    W = 1.0 + avg_ext_fo / (avg_front_fo + 1.0)

    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0)
    )

    return H