def init_mapping(self):
    """
    Hyperbolic Embedding Alignment Placement.

    Embeds the logical interaction graph and physical coupling graph into
    hyperbolic space (Poincaré disk model), then matches logical to physical
    qubits by minimizing hyperbolic distances via the Hungarian algorithm.

    Steps:
    1. Build weighted logical interaction graph with temporal decay.
    2. Compute Poincaré disk embeddings for both graphs via gradient descent
       on stress minimization in hyperbolic space.
    3. Build cost matrix of hyperbolic distances between all embedding pairs.
    4. Solve assignment via Hungarian algorithm.
    5. Refine with local pairwise-swap hill climbing.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    num_q = self.num_qubits
    gates_list = list(self.access.items())

    # Identify logical qubits actually used in the circuit
    logical_qubit_set = set()
    for _, qubits in gates_list:
        for q in qubits:
            logical_qubit_set.add(q)
    logical_qubits = sorted(logical_qubit_set)
    n_logical = len(logical_qubits)

    # Trivial fallback
    if n_logical <= 1:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        return

    # ── Step 1: Build weighted logical interaction graph with temporal decay ──
    total_gates = len(gates_list)
    interaction_weights = defaultdict(float)
    for idx, (gate_id, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            # Temporal decay: later gates weighted more (closer to execution)
            decay = 0.5 + 0.5 * (idx / max(total_gates - 1, 1))
            interaction_weights[key] += decay

    # Build logical graph adjacency and BFS distances
    log_adj = defaultdict(set)
    for (a, b) in interaction_weights:
        log_adj[a].add(b)
        log_adj[b].add(a)

    def bfs_distances(adj, nodes, start):
        dist = {n: float('inf') for n in nodes}
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]; head += 1
            for nb in adj.get(cur, []):
                if nb in dist and dist[nb] == float('inf'):
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        return dist

    # Logical graph distance matrix
    log_dist = np.full((n_logical, n_logical), float('inf'))
    for i, q in enumerate(logical_qubits):
        d = bfs_distances(log_adj, logical_qubit_set, q)
        for j, r in enumerate(logical_qubits):
            log_dist[i][j] = d.get(r, float('inf'))

    # Replace inf with diameter + 2
    finite_mask = np.isfinite(log_dist)
    max_finite = np.max(log_dist[finite_mask]) if np.any(finite_mask) else 1
    log_dist[~finite_mask] = max_finite + 2

    # Physical graph distance matrix
    phys_nodes = sorted(self.backend.keys())
    n_physical = len(phys_nodes)
    phys_dist = np.full((n_physical, n_physical), float('inf'))
    for i, p in enumerate(phys_nodes):
        d = bfs_distances(self.backend, set(phys_nodes), p)
        for j, r in enumerate(phys_nodes):
            phys_dist[i][j] = d.get(r, float('inf'))
    finite_mask_p = np.isfinite(phys_dist)
    max_finite_p = np.max(phys_dist[finite_mask_p]) if np.any(finite_mask_p) else 1
    phys_dist[~finite_mask_p] = max_finite_p + 2

    # ── Step 2: Poincaré disk embedding via gradient descent ──

    def hyp_distance(u, v):
        """Poincaré disk distance between points u and v."""
        diff = u - v
        diff_sq = np.sum(diff ** 2, axis=-1)
        norm_u_sq = np.sum(u ** 2, axis=-1)
        norm_v_sq = np.sum(v ** 2, axis=-1)
        denom = (1 - norm_u_sq) * (1 - norm_v_sq)
        denom = np.maximum(denom, 1e-10)
        arg = 1 + 2 * diff_sq / denom
        arg = np.maximum(arg, 1.0 + 1e-10)
        return np.arccosh(arg)

    def project_to_disk(X, max_norm=0.95):
        """Project points back into the Poincaré disk."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        mask = norms > max_norm
        X = np.where(mask, X * max_norm / (norms + 1e-12), X)
        return X

    def poincare_embed(target_dist, n_points, dim=2, n_iter=300, lr=0.01):
        """
        Embed n_points into the Poincaré disk by stress minimization.
        target_dist: n_points x n_points distance matrix (graph distances).
        Returns: n_points x dim array of Poincaré disk coordinates.
        """
        rng = np.random.RandomState(42)
        X = rng.uniform(-0.3, 0.3, (n_points, dim))
        X = project_to_disk(X)

        max_target = np.max(target_dist)
        if max_target > 0:
            scale = 3.5 / max_target
        else:
            scale = 1.0
        scaled_target = target_dist * scale

        for iteration in range(n_iter):
            current_lr = lr * (1 - 0.5 * iteration / n_iter)

            grad = np.zeros_like(X)
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    d_hyp = hyp_distance(X[i], X[j])
                    target_d = scaled_target[i, j]
                    error = d_hyp - target_d

                    direction = X[i] - X[j]
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        direction = rng.randn(dim) * 0.01
                        dir_norm = np.linalg.norm(direction)
                    direction /= dir_norm

                    # Riemannian correction: (1 - ||x||^2)^2 / 4
                    conf_i = (1 - np.sum(X[i] ** 2)) ** 2 / 4.0
                    conf_j = (1 - np.sum(X[j] ** 2)) ** 2 / 4.0

                    grad[i] += error * direction * conf_i
                    grad[j] -= error * direction * conf_j

            X -= current_lr * grad
            X = project_to_disk(X)

        return X

    log_embedding = poincare_embed(log_dist, n_logical, dim=2, n_iter=200, lr=0.02)
    phys_embedding = poincare_embed(phys_dist, n_physical, dim=2, n_iter=200, lr=0.02)

    # ── Step 3: Build cost matrix C[i,j] = hyperbolic distance ──
    cost_matrix = np.zeros((n_logical, n_physical))
    for i in range(n_logical):
        for j in range(n_physical):
            cost_matrix[i, j] = hyp_distance(log_embedding[i], phys_embedding[j])

    # ── Step 4: Solve assignment via Hungarian algorithm ──
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    lq_to_phys = {}
    for r, c in zip(row_ind, col_ind):
        lq = logical_qubits[r]
        pq = phys_nodes[c]
        lq_to_phys[lq] = pq

    # ── Step 5: Refine with local pairwise-swap hill climbing ──

    def compute_total_cost(mapping):
        total = 0.0
        for (q1, q2), w in interaction_weights.items():
            p1 = mapping[q1]
            p2 = mapping[q2]
            if p1 < len(self.distance_matrix) and p2 < len(self.distance_matrix):
                total += w * self.distance_matrix[p1][p2]
        return total

    # Build full bijective mapping via swap-based insertion
    mapping_dict = list(range(num_q))
    reverse_mapping_dict = list(range(num_q))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    # Hill-climbing: pairwise swap improvement
    improved = True
    max_rounds = 50
    round_count = 0
    current_cost = compute_total_cost(mapping_dict)

    while improved and round_count < max_rounds:
        improved = False
        round_count += 1
        for i in range(len(logical_qubits)):
            if improved:
                break
            for j in range(i + 1, len(logical_qubits)):
                lq_i = logical_qubits[i]
                lq_j = logical_qubits[j]
                p_i = mapping_dict[lq_i]
                p_j = mapping_dict[lq_j]

                # Try swap
                mapping_dict[lq_i] = p_j
                mapping_dict[lq_j] = p_i
                reverse_mapping_dict[p_i] = lq_j
                reverse_mapping_dict[p_j] = lq_i

                new_cost = compute_total_cost(mapping_dict)
                if new_cost < current_cost - 1e-12:
                    current_cost = new_cost
                    improved = True
                    break
                else:
                    # Revert
                    mapping_dict[lq_i] = p_i
                    mapping_dict[lq_j] = p_j
                    reverse_mapping_dict[p_i] = lq_i
                    reverse_mapping_dict[p_j] = lq_j

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict