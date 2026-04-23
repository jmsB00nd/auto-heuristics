def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits
    K = 3  # number of diffusion steps

    # ---- collect active logical qubits and logical interaction graph ----
    logical_adj = defaultdict(lambda: defaultdict(float))
    active_logical = set()
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits
            if a == b:
                continue
            logical_adj[a][b] += 1.0
            logical_adj[b][a] += 1.0
            active_logical.add(a)
            active_logical.add(b)

    # ---- physical graph from backend_connections ----
    physical_adj = defaultdict(lambda: defaultdict(float))
    for (u, v) in self.backend_connections:
        if u == v:
            continue
        physical_adj[u][v] += 1.0
        physical_adj[v][u] += 1.0

    physical_nodes = list(range(N))

    def diffusion_signature(node, adj, universe_size):
        # row-stochastic K-step diffusion starting from one-hot at node.
        # Use a dict-based sparse propagation, then return sorted probability vector.
        prob = {node: 1.0}
        for _ in range(K):
            nxt = defaultdict(float)
            for u, p in prob.items():
                neighbors = adj.get(u, {})
                deg = sum(neighbors.values())
                if deg <= 0:
                    # self-loop: stay in place (lazy walk fallback)
                    nxt[u] += p
                else:
                    for v, w in neighbors.items():
                        nxt[v] += p * (w / deg)
            prob = nxt
        vec = sorted(prob.values(), reverse=True)
        # pad/truncate to universe_size
        if len(vec) < universe_size:
            vec = vec + [0.0] * (universe_size - len(vec))
        else:
            vec = vec[:universe_size]
        return np.asarray(vec, dtype=float)

    sig_len = max(N, 1)
    active_logical_list = sorted(active_logical)

    # ---- build signatures ----
    log_sigs = {q: diffusion_signature(q, logical_adj, sig_len) for q in active_logical_list}
    phys_sigs = {p: diffusion_signature(p, physical_adj, sig_len) for p in physical_nodes}

    # ---- defaults: identity mapping fallback ----
    mapping_dict = list(range(N))
    reverse_mapping_dict = list(range(N))

    used_physical = set()
    assigned_logical = set()

    if active_logical_list and physical_nodes:
        L = len(active_logical_list)
        P = len(physical_nodes)
        size = max(L, P)
        BIG = 1e9
        cost = np.full((size, size), BIG, dtype=float)
        for i, lq in enumerate(active_logical_list):
            lv = log_sigs[lq]
            for j, pq in enumerate(physical_nodes):
                pv = phys_sigs[pq]
                diff = lv - pv
                cost[i, j] = float(np.dot(diff, diff))

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if r < L and c < P:
                    lq = active_logical_list[r]
                    pq = physical_nodes[c]
                    mapping_dict[lq] = pq
                    reverse_mapping_dict[pq] = lq
                    used_physical.add(pq)
                    assigned_logical.add(lq)
        except Exception:
            # greedy fallback
            order = sorted(range(L), key=lambda i: -float(np.sum(log_sigs[active_logical_list[i]])))
            taken = set()
            for i in order:
                lq = active_logical_list[i]
                best_j, best_d = None, float('inf')
                for j, pq in enumerate(physical_nodes):
                    if pq in taken:
                        continue
                    d = float(np.sum((log_sigs[lq] - phys_sigs[pq]) ** 2))
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j is not None:
                    pq = physical_nodes[best_j]
                    mapping_dict[lq] = pq
                    reverse_mapping_dict[pq] = lq
                    taken.add(pq)
                    used_physical.add(pq)
                    assigned_logical.add(lq)

    # ---- fill remaining logical qubits with leftover physical qubits ----
    remaining_phys = [p for p in physical_nodes if p not in used_physical]
    rp_iter = iter(remaining_phys)
    for lq in range(N):
        if lq in assigned_logical:
            continue
        # try to keep identity if free, else take next leftover
        if mapping_dict[lq] == lq and lq not in used_physical:
            used_physical.add(lq)
            reverse_mapping_dict[lq] = lq
            assigned_logical.add(lq)
            continue
        try:
            pq = next(rp_iter)
            while pq in used_physical:
                pq = next(rp_iter)
            mapping_dict[lq] = pq
            reverse_mapping_dict[pq] = lq
            used_physical.add(pq)
            assigned_logical.add(lq)
        except StopIteration:
            break

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if getattr(self, "use_isl", False):
        try:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        except Exception:
            pass

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)