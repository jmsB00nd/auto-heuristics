def qlosure_poly_heuristic(self, swap_gate):
    # Cache shortest-path counts on the topology (topology never changes during routing)
    if not hasattr(self, '_spdc_cache'):
        self._spdc_cache = {}

    def count_shortest_paths(src, tgt):
        """BFS count of distinct shortest paths from src to tgt on the backend graph."""
        key = (min(src, tgt), max(src, tgt))
        if key in self._spdc_cache:
            return self._spdc_cache[key]
        if src == tgt:
            self._spdc_cache[key] = 1
            return 1
        target_dist = int(self.distance_matrix[src][tgt])
        dist  = {src: 0}
        count = {src: 1}
        queue = [src]
        head  = 0
        while head < len(queue):
            node = queue[head]; head += 1
            if dist[node] >= target_dist:
                break
            for nb in self.backend[node]:
                if nb not in dist:
                    dist[nb]  = dist[node] + 1
                    count[nb] = count[node]
                    queue.append(nb)
                elif dist[nb] == dist[node] + 1:
                    count[nb] += count[node]
        result = count.get(tgt, 1)
        self._spdc_cache[key] = result
        return result

    W = 1
    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front layer: brittle pairs (few shortest paths) get amplified cost ---
    f_distance = 0
    for g in self.front_layer:
        q1, q2   = self.access2q[g]
        Q1, Q2   = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d        = self.distance_matrix[Q1][Q2]
        n_paths  = count_shortest_paths(Q1, Q2)
        # log-dampened diversity: log(1+1)≈0.69 (brittle) vs log(1+10)≈2.40 (flexible)
        diversity = math.log(1 + n_paths)
        deps      = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * d / diversity

    # --- Extended layer: same brittleness weighting + depth-decay ---
    e_distance = 0
    for g in self.extended_layer:
        q1, q2       = self.access2q[g]
        Q1, Q2       = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d            = self.distance_matrix[Q1][Q2]
        n_paths      = count_shortest_paths(Q1, Q2)
        diversity    = math.log(1 + n_paths)
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps         = self.dag_dependencies_count[g]
        e_distance   += (deps + 1) * (d / diversity) / layer_factor

    H = max_decay * (
        f_distance / front_layer_size
        + W * ((e_distance / extended_layer_size) if extended_layer_size else 0)
    )
    return H