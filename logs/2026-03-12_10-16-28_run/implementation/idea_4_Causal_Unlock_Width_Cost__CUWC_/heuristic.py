def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    if front_layer_size == 0:
        return 0.0

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Causal Unlock Width Cost (CUWC) ---
    #
    # unlock_width(g): number of direct successors in the full DAG that will
    # become *immediately* executable the moment g is executed — i.e., gates
    # for which g is their sole remaining unresolved predecessor.
    #
    # These gates are "activation keys": resolving g triggers a cascade of
    # newly executable operations. The more gates g unlocks, the more urgent
    # it is to route it quickly.
    #
    # Implementation note: self.dag_predecessors_full is dynamically updated
    # throughout execution (predecessors are discarded as gates execute), so
    # len(dag_predecessors_full[s]) gives the *current* remaining blocker count.
    # Since g is still in the front_layer (not yet executed), g is still present
    # in dag_predecessors_full[s] for all its direct successors s, making
    # len == 1 a reliable signal that g is the sole remaining blocker for s.

    def unlock_width(g):
        width = 0
        for s in self.dag_full.get(g, set()):
            if len(self.dag_predecessors_full.get(s, set())) == 1:
                width += 1
        return width

    # Front layer: gates with high unlock width are activation keys.
    # Scale their distance contribution by (1 + unlock_width) so that
    # SWAPs bringing high-unlock-width gates to adjacency are prioritised.
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        uw = unlock_width(g)
        f_cost += (1.0 + uw) * dist

    # Extended layer: same unlock-width amplification, discounted by lookahead
    # depth so that shallow extended gates with high unlock potential dominate.
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        uw = unlock_width(g)
        e_cost += (1.0 + uw) * dist / layer_factor

    W = 1.0
    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )
    return H