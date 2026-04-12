def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

    # ---------------------------------------------------------------
    # Step 1: Build distance matrix D (n_phys x n_phys)
    # ---------------------------------------------------------------
    D = np.zeros((n_phys, n_phys))
    for i, p1 in enumerate(physical_qubits):
        for j, p2 in enumerate(physical_qubits):
            D[i, j] = self.distance_matrix[p1][p2]

    # ---------------------------------------------------------------
    # Step 2: Build DAG and topological rank for temporal decay
    # ---------------------------------------------------------------
    schedule = sorted(self.access.keys())
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in schedule:
        write_qubits = self.write_dict.get(node, [])
        read_qubits = [q for q in self.access[node] if q not in write_qubits]
        for q in read_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            active_readers[q].add(node)
        for q in write_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            for r in active_readers.get(q, set()):
                if r != node:
                    successors[r].add(node)
                    predecessors[node].add(r)
            active_readers[q].clear()
            latest_writer[q] = node

    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_rank = {}
    rank = 0
    while queue:
        g = queue.popleft()
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    total_gates = max(rank, 1)

    # ---------------------------------------------------------------
    # Step 3: Build temporally-decayed interaction matrix W (n_phys x n_phys)
    # ---------------------------------------------------------------
    alpha = 2.5
    W = np.zeros((n_phys, n_phys))
    logical_qubits_set = set()

    # Map logical qubits to indices in [0, n_phys)
    # Physical qubits are already indexed by their position in physical_qubits
    # Logical qubits: we use their raw IDs directly as matrix indices (they are < num_q)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha * r / total_gates)
            if q1 < n_phys and q2 < n_phys:
                W[q1, q2] += w
                W[q2, q1] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    # ---------------------------------------------------------------
    # Step 4: QAP cost function: trace(W @ P @ D @ P^T)
    # P is a permutation matrix: P[i,j] = 1 means logical i -> physical j
    # ---------------------------------------------------------------
    def qap_cost(perm):
        """Compute trace(W P D P^T) for a given permutation."""
        cost = 0.0
        for i in range(n_phys):
            for j in range(n_phys):
                if W[i, j] != 0:
                    cost += W[i, j] * D[perm[i], perm[j]]
        return cost

    def perm_to_matrix(perm):
        P = np.zeros((n_phys, n_phys))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        return P

    def frank_wolfe_qap(P_init, num_iters=25):
        """Run Frank-Wolfe on QAP: min trace(W P D P^T) over doubly-stochastic P."""
        P = P_init.copy()
        for k in range(num_iters):
            # Gradient: dF/dP = W P D + W^T P D^T = 2 W P D (since W, D symmetric)
            grad = W @ P @ D + W.T @ P @ D.T
            # Linear minimization: find permutation Q minimizing trace(grad^T Q)
            # This is a linear assignment problem
            row_ind, col_ind = linear_sum_assignment(grad)
            Q = np.zeros((n_phys, n_phys))
            Q[row_ind, col_ind] = 1.0
            # Step size
            gamma = 2.0 / (k + 2)
            P = (1 - gamma) * P + gamma * Q
        # Project final P to nearest permutation
        row_ind, col_ind = linear_sum_assignment(-P)
        return list(col_ind)

    # ---------------------------------------------------------------
    # Step 5: Generate M=5 diverse seed permutations
    # ---------------------------------------------------------------

    # Seed 1: Identity permutation
    seed_identity = list(range(n_phys))

    # Seed 2: Gravity placement permutation
    def gravity_seed():
        # MDS embedding of physical qubits
        D_sq = D ** 2
        n = n_phys
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D_sq @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        idx_sorted = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sorted]
        eigvecs = eigvecs[:, idx_sorted]
        coords_2d = np.zeros((n, 2))
        for dim in range(min(2, n)):
            val = max(eigvals[dim], 0.0)
            coords_2d[:, dim] = eigvecs[:, dim] * np.sqrt(val)

        phys_pos = {i: coords_2d[i] for i in range(n)}
        centroid = np.mean(coords_2d, axis=0)

        # Build logical degree from W
        logical_degree = defaultdict(float)
        logical_neighbors = defaultdict(dict)
        for i in range(n_phys):
            for j in range(i + 1, n_phys):
                if W[i, j] > 0:
                    logical_neighbors[i][j] = W[i, j]
                    logical_neighbors[j][i] = W[i, j]
                    logical_degree[i] += W[i, j]
                    logical_degree[j] += W[i, j]

        logical_qubits = sorted(logical_qubits_set)
        used_physical = set()
        perm = list(range(n_phys))  # default identity

        def nearest_free(target_pos):
            best_pq = None
            best_dist = float('inf')
            for pq in range(n_phys):
                if pq not in used_physical:
                    d = np.linalg.norm(phys_pos[pq] - target_pos)
                    if d < best_dist:
                        best_dist = d
                        best_pq = pq
            return best_pq

        if logical_qubits:
            start_lq = max(logical_qubits, key=lambda q: logical_degree.get(q, 0))
            start_pq = nearest_free(centroid)
            perm[start_lq] = start_pq
            used_physical.add(start_pq)
            placed = {start_lq}
            remaining = set(logical_qubits) - placed

            while remaining:
                best_lq = max(remaining, key=lambda lq: sum(
                    logical_neighbors[lq].get(plq, 0.0) for plq in placed))
                neighbors_placed = {plq: logical_neighbors[best_lq].get(plq, 0.0)
                                    for plq in placed if plq in logical_neighbors[best_lq]}
                if neighbors_placed:
                    total_w = sum(neighbors_placed.values())
                    gravity = np.zeros(2)
                    for plq, wt in neighbors_placed.items():
                        gravity += wt * phys_pos[perm[plq]]
                    gravity /= total_w
                else:
                    gravity = centroid
                pq = nearest_free(gravity)
                perm[best_lq] = pq
                used_physical.add(pq)
                placed.add(best_lq)
                remaining.discard(best_lq)

            # Fill remaining
            unmapped = [q for q in range(n_phys) if q not in placed and q not in logical_qubits_set]
            free = [pq for pq in range(n_phys) if pq not in used_physical]
            for lq, pq in zip(unmapped, free):
                perm[lq] = pq

        return perm

    seed_gravity = gravity_seed()

    # Seed 3: Random permutation
    import random
    seed_random = list(range(n_phys))
    random.shuffle(seed_random)

    # Seed 4: Reverse Cuthill-McKee permutation
    def rcm_seed():
        def rcm_ordering(adj, nodes):
            if not nodes:
                return []
            node_set = set(nodes)
            deg = {v: len([u for u in adj.get(v, []) if u in node_set]) for v in nodes}
            start = min(nodes, key=lambda v: deg[v])
            visited = set()
            order = []
            q = deque([start])
            visited.add(start)
            while q:
                v = q.popleft()
                order.append(v)
                neighbors = sorted([u for u in adj.get(v, []) if u in node_set and u not in visited],
                                   key=lambda u: deg[u])
                for u in neighbors:
                    visited.add(u)
                    q.append(u)
            # Add any disconnected nodes
            for v in nodes:
                if v not in visited:
                    order.append(v)
            return list(reversed(order))

        # Hardware RCM
        hw_adj = {i: [physical_qubits.index(nb) for nb in self.backend[physical_qubits[i]] if nb in physical_qubits]
                  for i in range(n_phys)}
        hw_rcm = rcm_ordering(hw_adj, list(range(n_phys)))

        # Logical RCM from interaction graph
        log_adj = defaultdict(list)
        for i in range(n_phys):
            for j in range(n_phys):
                if W[i, j] > 0 and i != j:
                    log_adj[i].append(j)
        log_nodes = sorted(logical_qubits_set) if logical_qubits_set else list(range(n_phys))
        log_rcm = rcm_ordering(log_adj, log_nodes)

        # Map k-th logical in log_rcm to k-th physical in hw_rcm
        perm = list(range(n_phys))
        used = set()
        for k in range(min(len(log_rcm), len(hw_rcm))):
            perm[log_rcm[k]] = hw_rcm[k]
            used.add(hw_rcm[k])
        # Fill unmapped
        unmapped = [q for q in range(n_phys) if q not in set(log_rcm)]
        free = [pq for pq in range(n_phys) if pq not in used]
        for lq, pq in zip(unmapped, free):
            perm[lq] = pq
        return perm

    seed_rcm = rcm_seed()

    # Seed 5: Spectral embedding permutation
    def spectral_seed():
        # Laplacian of W (logical interaction graph)
        W_diag = np.diag(W.sum(axis=1))
        L_log = W_diag - W

        # Laplacian of hardware adjacency
        A_hw = np.zeros((n_phys, n_phys))
        for i in range(n_phys):
            for nb in self.backend.get(physical_qubits[i], []):
                if nb in physical_qubits:
                    j = physical_qubits.index(nb)
                    A_hw[i, j] = 1.0
        D_hw = np.diag(A_hw.sum(axis=1))
        L_hw = D_hw - A_hw

        # Compute Fiedler vectors (2nd and 3rd smallest eigenvectors)
        k = min(3, n_phys)
        eigvals_log, eigvecs_log = np.linalg.eigh(L_log)
        eigvals_hw, eigvecs_hw = np.linalg.eigh(L_hw)

        # Use columns 1..k-1 (skip trivial constant eigenvector)
        embed_log = eigvecs_log[:, 1:k]
        embed_hw = eigvecs_hw[:, 1:k]

        # Handle sign ambiguity
        for col in range(embed_log.shape[1]):
            if np.sum(embed_log[:, col]) < 0:
                embed_log[:, col] *= -1
            if np.sum(embed_hw[:, col]) < 0:
                embed_hw[:, col] *= -1

        # Normalize
        for col in range(embed_log.shape[1]):
            norm_l = np.linalg.norm(embed_log[:, col])
            norm_h = np.linalg.norm(embed_hw[:, col])
            if norm_l > 1e-12:
                embed_log[:, col] /= norm_l
            if norm_h > 1e-12:
                embed_hw[:, col] /= norm_h

        # Cost matrix: pairwise distances between embeddings
        cost = np.zeros((n_phys, n_phys))
        for i in range(n_phys):
            for j in range(n_phys):
                cost[i, j] = np.linalg.norm(embed_log[i] - embed_hw[j])

        row_ind, col_ind = linear_sum_assignment(cost)
        return list(col_ind)

    seed_spectral = spectral_seed()

    # ---------------------------------------------------------------
    # Step 6: Run Frank-Wolfe from each seed and pick best
    # ---------------------------------------------------------------
    seeds = [seed_identity, seed_gravity, seed_random, seed_rcm, seed_spectral]
    best_perm = None
    best_cost = float('inf')

    for seed_perm in seeds:
        P_init = perm_to_matrix(seed_perm)
        result_perm = frank_wolfe_qap(P_init, num_iters=25)
        cost = qap_cost(result_perm)
        if cost < best_cost:
            best_cost = cost
            best_perm = result_perm

    # ---------------------------------------------------------------
    # Step 7: Build mapping_dict and reverse_mapping_dict
    # ---------------------------------------------------------------
    # best_perm[i] = physical qubit index for logical qubit i
    # Map back to actual physical qubit IDs
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q

    for i in range(n_phys):
        pq = physical_qubits[best_perm[i]]
        mapping_dict[i] = pq
        reverse_mapping_dict[pq] = i

    # Fill any remaining unmapped qubits (if num_q > n_phys)
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)