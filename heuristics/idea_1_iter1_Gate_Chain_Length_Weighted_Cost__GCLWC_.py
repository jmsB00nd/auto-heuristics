# Idea: Gate Chain Length Weighted Cost (GCLWC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on multiplier_n75__1308CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    W = 1
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Step 1: Collect all gates reachable from front+extended via dag2q (2q-successors) ---
    relevant_gates = set(self.front_layer) | set(self.extended_layer)
    reachable = set()
    stack = list(relevant_gates)
    while stack:
        g = stack.pop()
        if g in reachable:
            continue
        reachable.add(g)
        for s in self.dag2q.get(g, set()):
            if s not in reachable:
                stack.append(s)

    # --- Step 2: Iterative post-order DFS → topological order (sinks first, sources last) ---
    topo_order = []
    visited = set()
    for start in reachable:
        if start in visited:
            continue
        dfs_stack = [(start, False)]
        while dfs_stack:
            node, processed = dfs_stack.pop()
            if processed:
                topo_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            dfs_stack.append((node, True))   # re-push to append after children
            for succ in self.dag2q.get(node, set()):
                if succ in reachable and succ not in visited:
                    dfs_stack.append((succ, False))

    # --- Step 3: Bottom-up chain computation (leaves first in topo_order) ---
    # chain(g) = 1 + max(chain(s) for s in 2q-successors), leaf = 1
    chain = {}
    for g in topo_order:
        succs = [s for s in self.dag2q.get(g, set()) if s in reachable]
        if not succs:
            chain[g] = 1
        else:
            chain[g] = 1 + max(chain.get(s, 1) for s in succs)

    # --- Step 4: Front layer — weighted by forward chain length ---
    f_distance = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        chain_weight = chain.get(g, 1)   # forward-looking: gates heading a long 2q-chain cost more
        f_distance += chain_weight * self.distance_matrix[Q1][Q2]

    # --- Step 5: Extended layer — chain weight discounted by lookahead depth ---
    e_distance = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        chain_weight = chain.get(g, 1)
        e_distance += chain_weight * self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size +
        W * ((e_distance / extended_layer_size) if extended_layer_size else 0)
    )

    return H