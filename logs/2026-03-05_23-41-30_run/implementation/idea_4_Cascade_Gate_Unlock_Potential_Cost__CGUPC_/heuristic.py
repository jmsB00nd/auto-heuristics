def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    if front_layer_size == 0:
        return 0.0

    # --- Cascade unlock scorer (2-hop look-ahead) ---
    # Level-1: successors of g that immediately enter front_layer when g executes.
    # Level-2: successors-of-successors that would further unlock assuming level-1 all execute.
    # The two levels are discounted differently to model diminishing cascade certainty.
    def cascade_score(g):
        level1 = set()
        for succ in self.dag_full.get(g, set()):
            if len(self.dag_predecessors_full.get(succ, set())) == 1:
                level1.add(succ)

        level2_count = 0
        for lvl1 in level1:
            for succ2 in self.dag_full.get(lvl1, set()):
                if succ2 in level1:
                    continue
                # succ2 unlocks if every remaining blocker is either g or a level-1 gate
                remaining = self.dag_predecessors_full.get(succ2, set()) - {g} - level1
                if len(remaining) == 0:
                    level2_count += 1

        # Direct unlocks count fully; indirect unlocks are discounted by 0.5
        return len(level1) + 0.5 * level2_count

    # --- Front layer: super-linear cascade-amplified distance ---
    # Rationale: a gate that cascades K unlocks saves ~K+1 routing decisions.
    # Super-linear exponent (^1.5) ensures high-cascade gates dominate the gradient,
    # breaking ties aggressively in favour of cascade-rich paths.
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        cas = cascade_score(g)
        f_cost += (1.0 + cas) ** 1.5 * dist

    f_cost /= front_layer_size

    # --- Extended layer: log-cascade with exponential depth decay ---
    # Rationale: cascade benefit diminishes with DAG depth (uncertainty grows),
    # modelled as log(1 + cas). Depth is penalised exponentially (not harmonically)
    # so the lookahead signal collapses faster, avoiding over-optimism.
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        layer_idx = self.extended_layer_index.get(g, 0) + 1
        cas = cascade_score(g)
        log_cas = math.log(1.0 + cas)
        depth_decay = math.exp(-0.5 * (layer_idx - 1))
        e_cost += (1.0 + log_cas) * dist * depth_decay

    e_cost = (e_cost / extended_layer_size) if extended_layer_size else 0.0

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    H = max_decay * (f_cost + e_cost)
    return H