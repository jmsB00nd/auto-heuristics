# Idea: Critical Path Gate Urgency Cost (CPGUC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on multiplier_n75__1308CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    # --- Step 1: Precompute critical path lengths via iterative post-order DFS ---
    # CP(g) = length of longest directed path from g to any sink in dag2q
    cp_cache = {}

    def compute_cp(start):
        """Iterative post-order DFS: ensures all successors are resolved before parent."""
        stack = [(start, False)]
        while stack:
            node, processed = stack.pop()
            if node in cp_cache:
                continue
            if processed:
                successors = self.dag2q.get(node, set())
                if not successors:
                    cp_cache[node] = 1
                else:
                    cp_cache[node] = 1 + max(cp_cache.get(s, 1) for s in successors)
            else:
                stack.append((node, True))
                for s in self.dag2q.get(node, set()):
                    if s not in cp_cache:
                        stack.append((s, False))

    all_gates = list(self.front_layer) + list(self.extended_layer)
    if not all_gates:
        return 0.0

    for g in all_gates:
        compute_cp(g)

    # --- Step 2: Normalize by the maximum CP among all relevant gates ---
    max_cp = max(cp_cache.get(g, 1) for g in all_gates)
    if max_cp == 0:
        max_cp = 1

    # --- Step 3: Accumulate weighted distances ---
    W = 1
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        # Critical-path urgency weight: gates deeper in the dependency chain score higher
        w_g = cp_cache.get(g, 1) / max_cp
        f_distance += w_g * self.distance_matrix[Q1][Q2]

    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        w_g = cp_cache.get(g, 1) / max_cp
        e_distance += w_g * self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size
        + W * ((e_distance / extended_layer_size) if extended_layer_size else 0)
    )

    return H