# Idea: Tensor Product Neighborhood Alignment Cost (TPNAC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on qugan_n71__72CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    W = 0.5
    front_layer_size  = max(len(self.front_layer), 1)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # ------------------------------------------------------------------ #
    # Core TPNAC metric                                                   #
    # ------------------------------------------------------------------ #
    def tpnac(Q1, Q2):
        if Q1 < 0 or Q2 < 0:
            return 0.0
        dist = self.distance_matrix[Q1][Q2]
        if dist == 0:
            return 0.0
        # |N(Q1) ∩ N(Q2)|  — set intersection of hardware neighbours
        common = len(self.backend[Q1] & self.backend[Q2])
        return dist / (common + 1)

    # ------------------------------------------------------------------ #
    # Front-layer cost                                                    #
    #   Weighted by criticality so bottleneck gates (many dependants)     #
    #   dominate the objective.                                           #
    # ------------------------------------------------------------------ #
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        criticality = self.dag_dependencies_count[g] + 1
        f_cost += criticality * tpnac(Q1, Q2)

    # ------------------------------------------------------------------ #
    # Extended-layer (lookahead) cost                                     #
    #   Discounted by lookahead depth so near-future gates carry more     #
    #   weight than distant future gates.                                 #
    # ------------------------------------------------------------------ #
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        depth       = self.extended_layer_index.get(g, 0) + 1
        criticality = self.dag_dependencies_count[g] + 1
        e_cost += criticality * tpnac(Q1, Q2) / depth

    # ------------------------------------------------------------------ #
    # Final cost                                                          #
    # ------------------------------------------------------------------ #
    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H