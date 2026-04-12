def init_mapping(self):
    """
    Graph Neural Network Embedding Alignment (GNN-Embed).

    Uses a lightweight, untrained random-weight 2-layer GNN to produce structural
    embeddings of both the logical interaction graph and the hardware graph.
    Matches embeddings via Hungarian assignment + local search refinement.
    """
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import math

    gates_list = list(self.access.items())

    # Identify logical qubits and build interaction graph
    logical_qubit_set = set()
    interaction_weight = defaultdict(float)

    # Forward pass for layer-based weighting
    qubit_ready = {}
    gate_start = []
    gate_finish = []
    for _, qubits in gates_list:
        es = max((qubit_ready.get(q, 0) for q in qubits), default=0)
        ef = es + 1
        gate_start.append(es)
        gate_finish.append(ef)
        for q in qubits:
            qubit_ready[q] = ef
            logical_qubit_set.add(q)

    total_depth = max(gate_finish, default=1)
    half_life = max(total_depth / 4.0, 4.0)

    for idx, (_, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_start[idx]
            w = math.exp(-layer * math.log(2) / half_life) + 0.05
            interaction_weight[key] += w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_lq = len(logical_qubits)
    n_pq = len(physical_qubits)
    lq_index = {lq: i for i, lq in enumerate(logical_qubits)}
    pq_index = {pq: i for i, pq in enumerate(physical_qubits)}

    # Build adjacency structures
    lq_adj = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        lq_adj[q1][q2] = w
        lq_adj[q2][q1] = w

    pq_set = set(physical_qubits)
    pq_adj = {pq: [nb for nb in self.backend.get(pq, []) if nb in pq_set]
              for pq in physical_qubits}

    # ---- GNN Parameters ----
    embed_dim = 16
    rng = np.random.RandomState(42)

    # ---- Step 1: Compute initial node features ----
    # Logical: [degree, total_interaction_weight]
    logical_features = np.zeros((n_lq, 2))
    for i, lq in enumerate(logical_qubits):
        neighbors = lq_adj[lq]
        logical_features[i, 0] = len(neighbors)
        logical_features[i, 1] = sum(neighbors.values())

    # Physical: [degree, closeness_centrality]
    physical_features = np.zeros((n_pq, 2))
    for i, pq in enumerate(physical_qubits):
        deg = len(pq_adj[pq])
        physical_features[i, 0] = deg
        centrality = sum(
            1.0 / self.distance_matrix[pq][other]
            for other in physical_qubits
            if other != pq and self.distance_matrix[pq][other] not in (0, float('inf'))
        )
        physical_features[i, 1] = centrality

    # Normalize features to [0, 1] range for stability
    for col in range(2):
        for feats in [logical_features, physical_features]:
            col_max = feats[:, col].max()
            col_min = feats[:, col].min()
            if col_max - col_min > 1e-12:
                feats[:, col] = (feats[:, col] - col_min) / (col_max - col_min)

    # ---- Step 2: Random GNN weights (2 layers) ----
    input_dim = 2
    hidden_dim = embed_dim
    W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    W2 = rng.randn(hidden_dim, embed_dim) * np.sqrt(2.0 / hidden_dim)

    # ---- Step 3: GNN message passing ----
    def gnn_forward(features, adj_func, n_nodes, W1, W2):
        """2-layer GNN with mean aggregation + ReLU."""
        h = features

        # Layer 1: aggregate + transform + ReLU
        agg1 = np.zeros((n_nodes, h.shape[1]))
        for i in range(n_nodes):
            neighbors, weights = adj_func(i)
            if neighbors:
                neighbor_feats = h[neighbors]
                if weights is not None:
                    w_arr = np.array(weights).reshape(-1, 1)
                    w_sum = w_arr.sum()
                    if w_sum > 1e-12:
                        agg1[i] = (neighbor_feats * w_arr).sum(axis=0) / w_sum
                    else:
                        agg1[i] = neighbor_feats.mean(axis=0)
                else:
                    agg1[i] = neighbor_feats.mean(axis=0)
            agg1[i] = (h[i] + agg1[i]) / 2.0

        h1 = np.maximum(0, agg1 @ W1)  # ReLU

        # Layer 2: aggregate + transform + ReLU
        agg2 = np.zeros((n_nodes, h1.shape[1]))
        for i in range(n_nodes):
            neighbors, weights = adj_func(i)
            if neighbors:
                neighbor_feats = h1[neighbors]
                if weights is not None:
                    w_arr = np.array(weights).reshape(-1, 1)
                    w_sum = w_arr.sum()
                    if w_sum > 1e-12:
                        agg2[i] = (neighbor_feats * w_arr).sum(axis=0) / w_sum
                    else:
                        agg2[i] = neighbor_feats.mean(axis=0)
                else:
                    agg2[i] = neighbor_feats.mean(axis=0)
            agg2[i] = (h1[i] + agg2[i]) / 2.0

        h2 = np.maximum(0, agg2 @ W2)  # ReLU
        return h2

    # Logical graph adjacency (weighted)
    def logical_adj(i):
        lq = logical_qubits[i]
        neighbors = lq_adj[lq]
        if not neighbors:
            return [], None
        nb_indices = []
        nb_weights = []
        for nb, w in neighbors.items():
            if nb in lq_index:
                nb_indices.append(lq_index[nb])
                nb_weights.append(w)
        return nb_indices, nb_weights

    # Physical graph adjacency (unweighted)
    def physical_adj(i):
        pq = physical_qubits[i]
        neighbors = pq_adj[pq]
        if not neighbors:
            return [], None
        nb_indices = [pq_index[nb] for nb in neighbors if nb in pq_index]
        return nb_indices, None

    logical_embeddings = gnn_forward(logical_features, logical_adj, n_lq, W1, W2)
    physical_embeddings = gnn_forward(physical_features, physical_adj, n_pq, W1, W2)

    # ---- Step 4: Normalize embeddings ----
    for emb in [logical_embeddings, physical_embeddings]:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        emb /= norms

    # ---- Step 5: Cost matrix via Euclidean distance ----
    cost_matrix = np.zeros((n_lq, n_pq))
    for i in range(n_lq):
        diff = physical_embeddings - logical_embeddings[i]
        cost_matrix[i] = np.sum(diff * diff, axis=1)

    # ---- Step 6: Hungarian assignment ----
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    gnn_map = {
        logical_qubits[r]: physical_qubits[c]
        for r, c in zip(row_ind, col_ind)
    }

    # ---- Step 7: Build full bijection ----
    def make_mapping(lq_phys):
        md = list(range(self.num_qubits))
        for lq, pq in lq_phys.items():
            md[lq] = pq
        assigned = set(lq_phys.values())
        remaining = [pq for pq in range(self.num_qubits) if pq not in assigned]
        ri = 0
        for lq in range(self.num_qubits):
            if lq not in lq_phys:
                md[lq] = remaining[ri]
                ri += 1
        rmd = list(range(self.num_qubits))
        for lq in range(self.num_qubits):
            rmd[md[lq]] = lq
        return md, rmd

    # ---- Step 8: QAP cost and local search utilities ----
    lq_combined = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        lq_combined[q1][q2] = w
        lq_combined[q2][q1] = w

    def qap_cost(md):
        return sum(
            w * self.distance_matrix[md[q1]][md[q2]]
            for (q1, q2), w in interaction_weight.items()
        )

    def swap_delta(p1, p2, md, rmd):
        a, b = rmd[p1], rmd[p2]
        delta = 0.0
        for nb, w in lq_combined[a].items():
            pp = md[nb]
            if pp == p2:
                continue
            delta += w * (self.distance_matrix[p2][pp] - self.distance_matrix[p1][pp])
        for nb, w in lq_combined[b].items():
            pp = md[nb]
            if pp == p1:
                continue
            delta += w * (self.distance_matrix[p1][pp] - self.distance_matrix[p2][pp])
        return delta

    def do_swap(p1, p2, md, rmd):
        a, b = rmd[p1], rmd[p2]
        md[a], md[b] = p2, p1
        rmd[p1], rmd[p2] = b, a

    lq_by_weight = sorted(
        logical_qubits,
        key=lambda lq: sum(lq_combined[lq].values()),
        reverse=True
    )

    pq_r2 = {}
    for pq in physical_qubits:
        nbrs = set(pq_adj[pq])
        for nb in pq_adj[pq]:
            nbrs.update(pq_adj[nb])
        nbrs.discard(pq)
        pq_r2[pq] = list(nbrs)

    # ---- Step 9: Local search refinement ----
    def local_search(md, rmd, max_iters=400):
        for iteration in range(max_iters):
            improved = False
            do_full = (iteration % 5 == 0)
            for lq in lq_by_weight:
                p1 = md[lq]
                best_d, best_p2 = -1e-9, -1
                for p2 in pq_r2.get(p1, []):
                    d = swap_delta(p1, p2, md, rmd)
                    if d < best_d:
                        best_d, best_p2 = d, p2
                if do_full:
                    r2_set = set(pq_r2.get(p1, []))
                    r2_set.add(p1)
                    for p2 in physical_qubits:
                        if p2 in r2_set:
                            continue
                        d = swap_delta(p1, p2, md, rmd)
                        if d < best_d:
                            best_d, best_p2 = d, p2
                if best_p2 != -1:
                    do_swap(p1, best_p2, md, rmd)
                    improved = True
            if not improved:
                break
        return md, rmd

    # ---- Step 10: Evaluate GNN seed + local search ----
    md, rmd = make_mapping(gnn_map)
    md, rmd = local_search(md, rmd)
    best_cost = qap_cost(md)
    best_md, best_rmd = md[:], rmd[:]

    # ---- Step 11: Additional random GNN seeds for robustness ----
    for seed in [7, 13, 99]:
        rng2 = np.random.RandomState(seed)
        W1_alt = rng2.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        W2_alt = rng2.randn(hidden_dim, embed_dim) * np.sqrt(2.0 / hidden_dim)

        le = gnn_forward(logical_features, logical_adj, n_lq, W1_alt, W2_alt)
        pe = gnn_forward(physical_features, physical_adj, n_pq, W1_alt, W2_alt)

        for emb in [le, pe]:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            emb /= norms

        cm = np.zeros((n_lq, n_pq))
        for i in range(n_lq):
            diff = pe - le[i]
            cm[i] = np.sum(diff * diff, axis=1)

        ri2, ci2 = linear_sum_assignment(cm)
        alt_map = {logical_qubits[r]: physical_qubits[c] for r, c in zip(ri2, ci2)}
        md2, rmd2 = make_mapping(alt_map)
        md2, rmd2 = local_search(md2, rmd2)
        c2 = qap_cost(md2)
        if c2 < best_cost:
            best_cost = c2
            best_md, best_rmd = md2[:], rmd2[:]

    # ---- Step 12: SA refinement ----
    import random as pyrandom
    pyrandom_rng = pyrandom.Random(42)

    md = best_md[:]
    rmd = best_rmd[:]
    current_cost = best_cost

    active_pqs = list(set(md[lq] for lq in logical_qubits if lq_combined[lq]))
    if len(active_pqs) < 2:
        active_pqs = physical_qubits

    n_sa = max(3000, n_lq * 200)
    T_start = max(current_cost * 0.05, 0.5)
    T_end = max(current_cost * 0.00005, 1e-4)
    alpha = (T_end / T_start) ** (1.0 / n_sa) if n_sa > 0 else 1.0
    T = T_start

    for _ in range(n_sa):
        p1, p2 = pyrandom_rng.sample(active_pqs, 2)
        delta = swap_delta(p1, p2, md, rmd)
        if delta < 0 or (T > 1e-9 and pyrandom_rng.random() < math.exp(-delta / T)):
            do_swap(p1, p2, md, rmd)
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_md = md[:]
                best_rmd = rmd[:]
        T *= alpha

    # Final local search
    best_md, best_rmd = local_search(best_md, best_rmd, max_iters=500)

    self.mapping_dict = best_md
    self.reverse_mapping_dict = best_rmd

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)