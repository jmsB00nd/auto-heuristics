# Idea: Congestion-Aware Geodesic Routing Cost (CAGRC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    p0, p1 = swap_gate[0], swap_gate[1]
    swap_edge = (min(p0, p1), max(p0, p1))

    front_layer_size  = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # ------------------------------------------------------------------ #
    # Step 1: Greedy shortest-path edge enumeration                        #
    #   Uses distance_matrix to walk from src toward dst one hop at a time.#
    #   Each step picks any neighbor that is exactly 1 closer to dst.      #
    # ------------------------------------------------------------------ #
    def get_path_edges(src, dst):
        if src == dst or src < 0 or dst < 0:
            return []
        edges = []
        cur = src
        while cur != dst:
            d = self.distance_matrix[cur][dst]
            for nb in self.backend[cur]:
                if self.distance_matrix[nb][dst] == d - 1:
                    edges.append((min(cur, nb), max(cur, nb)))
                    cur = nb
                    break
            else:
                break  # disconnected (safety guard)
        return edges

    # ------------------------------------------------------------------ #
    # Step 2: Build edge congestion map                                    #
    #   Each gate contributes weight = (criticality + 1) to every edge    #
    #   along its optimal routing path. Extended-layer gates are           #
    #   discounted by their lookahead depth to mirror temporal relevance.  #
    # ------------------------------------------------------------------ #
    edge_congestion = {}

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        w = float(self.dag_dependencies_count[g] + 1)
        for e in get_path_edges(Q1, Q2):
            edge_congestion[e] = edge_congestion.get(e, 0.0) + w

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        depth = self.extended_layer_index.get(g, 0) + 1
        w = float(self.dag_dependencies_count[g] + 1) / depth
        for e in get_path_edges(Q1, Q2):
            edge_congestion[e] = edge_congestion.get(e, 0.0) + w

    # ------------------------------------------------------------------ #
    # Step 3: Relative congestion of the candidate SWAP edge               #
    #   Normalise by mean edge congestion across the entire hardware so    #
    #   the penalty is dimensionless and topology-independent.             #
    #   relative_congestion > 1  ↔  this edge is a bottleneck.            #
    # ------------------------------------------------------------------ #
    swap_congestion  = edge_congestion.get(swap_edge, 0.0)
    total_congestion = sum(edge_congestion.values()) if edge_congestion else 1.0
    num_edges        = len(self.backend_connections) if self.backend_connections else 1
    mean_congestion  = total_congestion / num_edges
    relative_congestion = swap_congestion / (mean_congestion + 1e-9)

    # ------------------------------------------------------------------ #
    # Step 4: Distance terms (ensure gates still make forward progress)   #
    # ------------------------------------------------------------------ #
    f_dist = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 >= 0 and Q2 >= 0:
            f_dist += self.distance_matrix[Q1][Q2]
    f_dist /= (front_layer_size or 1)

    e_dist = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 >= 0 and Q2 >= 0:
            layer_idx = self.extended_layer_index.get(g, 0) + 1
            e_dist += self.distance_matrix[Q1][Q2] / layer_idx
    e_dist /= (extended_layer_size or 1)

    # ------------------------------------------------------------------ #
    # Step 5: Qubit health decay                                           #
    # ------------------------------------------------------------------ #
    max_decay = max(self.decay_parameter[p0], self.decay_parameter[p1])

    # ------------------------------------------------------------------ #
    # Step 6: CAGRC cost                                                   #
    #                                                                      #
    #   cost = decay × ( f_dist                                           #
    #                   + 0.5 × e_dist          (lookahead)               #
    #                   + 1.0 × rel_congestion  (bottleneck penalty) )    #
    #                                                                      #
    #   The congestion term shifts the objective from purely local         #
    #   gate-distance minimisation toward global traffic-flow optimisation:#
    #   SWAPs on bottleneck edges (high relative congestion) are penalised #
    #   to prevent cascading future SWAP chains.                           #
    # ------------------------------------------------------------------ #
    W_ext  = 0.5
    W_cong = 1.0
    cost = max_decay * (f_dist + W_ext * e_dist + W_cong * relative_congestion)

    return cost