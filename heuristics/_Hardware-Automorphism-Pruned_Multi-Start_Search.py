def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)
    pq_set = set(physical_qubits)

    # ===================================================================
    # Phase 0: Compute automorphism orbits via iterative color refinement
    # ===================================================================
    # Assign initial colors based on degree
    color = {}
    for pq in physical_qubits:
        color[pq] = len(self.backend[pq])

    # Iterative refinement: refine colors based on neighbor color multisets
    for _ in range(n_phys):
        new_color = {}
        color_tuples = {}
        for pq in physical_qubits:
            neighbor_colors = tuple(sorted(color[nb] for nb in self.backend[pq]))
            color_tuples[pq] = (color[pq], neighbor_colors)

        # Re-label colors as integers
        unique_labels = {}
        label_counter = 0
        for pq in physical_qubits:
            ct = color_tuples[pq]
            if ct not in unique_labels:
                unique_labels[ct] = label_counter
                label_counter += 1
            new_color[pq] = unique_labels[ct]

        if all(new_color[pq] == color[pq] for pq in physical_qubits):
            break
        color = new_color

    # Group physical qubits into orbits (same final color = same orbit)
    orbits = defaultdict(list)
    for pq in physical_qubits:
        orbits[color[pq]].append(pq)

    # ===================================================================
    # Phase 0b: Build interaction graph and DAG for the circuit
    # ===================================================================
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

    # Topological rank via Kahn's algorithm
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

    # Interaction weights with temporal decay
    alpha = 2.5
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha * r / total_gates)
            interaction_weight[key] += w
            logical_degree[q1] += w
            logical_degree[q2] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)
    n_logical = len(logical_qubits)

    if n_logical == 0:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Build logical adjacency
    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    # ===================================================================
    # Phase 1: Identify diameter endpoints, pick orbit representatives
    # ===================================================================
    # Compute graph diameter and find all diameter endpoint pairs
    diameter = 0
    for i in physical_qubits:
        for j in physical_qubits:
            d = self.distance_matrix[i][j]
            if d != float('inf') and d > diameter:
                diameter = d

    # Collect diameter endpoints (individual nodes that participate)
    diameter_endpoints = set()
    for i in physical_qubits:
        for j in physical_qubits:
            if self.distance_matrix[i][j] == diameter:
                diameter_endpoints.add(i)

    # Select one representative per orbit among diameter endpoints
    orbit_representatives = []
    seen_orbits = set()
    for pq in physical_qubits:
        if pq in diameter_endpoints:
            orb = color[pq]
            if orb not in seen_orbits:
                seen_orbits.add(orb)
                orbit_representatives.append(pq)

    # If no diameter endpoints found, use centrality-based representatives
    if not orbit_representatives:
        seen_orbits = set()
        for pq in physical_qubits:
            orb = color[pq]
            if orb not in seen_orbits:
                seen_orbits.add(orb)
                orbit_representatives.append(pq)

    # Limit multi-start to reasonable number
    if len(orbit_representatives) > 12:
        # Pick representatives with diverse eccentricities
        ecc = {}
        for pq in orbit_representatives:
            ecc[pq] = max(self.distance_matrix[pq][j] for j in physical_qubits
                         if self.distance_matrix[pq][j] != float('inf'))
        orbit_representatives.sort(key=lambda x: -ecc[x])
        orbit_representatives = orbit_representatives[:12]

    # ===================================================================
    # Phase 2: Spectral partition expansion placement for each representative
    # ===================================================================
    # Build distance matrix for physical qubits (dense numpy)
    pq_idx = {pq: i for i, pq in enumerate(physical_qubits)}
    D_phys = np.zeros((n_phys, n_phys))
    for i, p1 in enumerate(physical_qubits):
        for j, p2 in enumerate(physical_qubits):
            D_phys[i, j] = self.distance_matrix[p1][p2]

    # Spectral embedding of hardware graph (Laplacian eigenvectors)
    adj_phys = np.zeros((n_phys, n_phys))
    for pq in physical_qubits:
        for nb in self.backend[pq]:
            adj_phys[pq_idx[pq], pq_idx[nb]] = 1.0

    deg_phys = np.diag(adj_phys.sum(axis=1))
    L_phys = deg_phys - adj_phys
    eigvals_p, eigvecs_p = np.linalg.eigh(L_phys)

    # Spectral embedding of logical interaction graph
    lq_idx = {lq: i for i, lq in enumerate(logical_qubits)}
    adj_log = np.zeros((n_logical, n_logical))
    for (q1, q2), w in interaction_weight.items():
        adj_log[lq_idx[q1], lq_idx[q2]] = w
        adj_log[lq_idx[q2], lq_idx[q1]] = w

    deg_log = np.diag(adj_log.sum(axis=1))
    L_log = deg_log - adj_log
    eigvals_l, eigvecs_l = np.linalg.eigh(L_log)

    # Use first k non-trivial eigenvectors for embedding
    k = min(6, n_logical - 1, n_phys - 1)
    k = max(k, 1)

    # Physical embedding: skip first (constant) eigenvector
    phys_embed = eigvecs_p[:, 1:1+k]
    log_embed = eigvecs_l[:, 1:1+k]

    # Pad if dimensions don't match
    if log_embed.shape[1] < k:
        pad = k - log_embed.shape[1]
        log_embed = np.hstack([log_embed, np.zeros((n_logical, pad))])
    if phys_embed.shape[1] < k:
        pad = k - phys_embed.shape[1]
        phys_embed = np.hstack([phys_embed, np.zeros((n_phys, pad))])

    def build_placement_from_anchor(anchor_pq):
        """Build a placement starting from anchor_pq using spectral alignment."""
        # Find the logical qubit with highest degree to anchor
        anchor_lq = max(logical_qubits, key=lambda q: logical_degree.get(q, 0))

        # Align spectral embeddings using Procrustes-like approach
        # Translate both embeddings so anchor is at origin
        phys_shifted = phys_embed - phys_embed[pq_idx[anchor_pq]]
        log_shifted = log_embed - log_embed[lq_idx[anchor_lq]]

        # Compute cost matrix: distance between each logical and physical embedding
        cost = np.zeros((n_logical, n_phys))
        for i in range(n_logical):
            for j in range(n_phys):
                cost[i, j] = np.linalg.norm(log_shifted[i] - phys_shifted[j])

        # Bias cost toward anchor assignment
        cost[lq_idx[anchor_lq], :] += 1e6
        cost[lq_idx[anchor_lq], pq_idx[anchor_pq]] = 0.0

        # Also bias toward nearby physical qubits for high-degree logical qubits
        for lq in logical_qubits:
            deg_bias = logical_degree.get(lq, 0)
            for j, pq in enumerate(physical_qubits):
                cost[lq_idx[lq], j] += 0.1 * deg_bias * self.distance_matrix[anchor_pq][pq] / max(diameter, 1)

        # Greedy assignment (Hungarian is O(n^3), greedy is faster for large backends)
        mapping = [-1] * num_q
        rev_mapping = [-1] * num_q
        used_phys = set()
        used_log = set()

        # Force anchor assignment
        mapping[anchor_lq] = anchor_pq
        rev_mapping[anchor_pq] = anchor_lq
        used_phys.add(anchor_pq)
        used_log.add(anchor_lq)

        # Assign remaining by sorting cost entries
        assignments = []
        for i, lq in enumerate(logical_qubits):
            if lq in used_log:
                continue
            for j, pq in enumerate(physical_qubits):
                assignments.append((cost[i, j], lq, pq))
        assignments.sort()

        for _, lq, pq in assignments:
            if lq in used_log or pq in used_phys:
                continue
            mapping[lq] = pq
            rev_mapping[pq] = lq
            used_phys.add(pq)
            used_log.add(lq)

        # Fill unmapped
        unmapped_log = [q for q in range(num_q) if mapping[q] == -1]
        free_phys = [pq for pq in range(num_q) if rev_mapping[pq] == -1]
        for lq, pq in zip(unmapped_log, free_phys):
            mapping[lq] = pq
            rev_mapping[pq] = lq

        return mapping, rev_mapping

    def evaluate_mapping(mapping):
        """Score a mapping by weighted distance of interacting qubit pairs."""
        total = 0.0
        for (q1, q2), w in interaction_weight.items():
            p1, p2 = mapping[q1], mapping[q2]
            total += w * self.distance_matrix[p1][p2]
        return total

    # ===================================================================
    # Run multi-start search across orbit representatives
    # ===================================================================
    best_score = float('inf')
    best_mapping = None
    best_rev = None

    for anchor in orbit_representatives:
        m, rm = build_placement_from_anchor(anchor)
        score = evaluate_mapping(m)
        if score < best_score:
            best_score = score
            best_mapping = m
            best_rev = rm

    # ===================================================================
    # Phase 3: 2-opt refinement on the best mapping
    # ===================================================================
    improved = True
    max_rounds = 5
    round_count = 0

    while improved and round_count < max_rounds:
        improved = False
        round_count += 1

        # Only consider swapping logical qubits that participate in 2q gates
        candidates = logical_qubits if len(logical_qubits) <= 200 else logical_qubits[:200]

        for i_idx in range(len(candidates)):
            for j_idx in range(i_idx + 1, len(candidates)):
                lq1 = candidates[i_idx]
                lq2 = candidates[j_idx]
                pq1 = best_mapping[lq1]
                pq2 = best_mapping[lq2]

                # Compute delta: only consider edges involving lq1 or lq2
                delta = 0.0
                for lq_other, w in logical_neighbors[lq1].items():
                    if lq_other == lq2:
                        continue
                    p_other = best_mapping[lq_other]
                    delta += w * (self.distance_matrix[pq2][p_other] - self.distance_matrix[pq1][p_other])

                for lq_other, w in logical_neighbors[lq2].items():
                    if lq_other == lq1:
                        continue
                    p_other = best_mapping[lq_other]
                    delta += w * (self.distance_matrix[pq1][p_other] - self.distance_matrix[pq2][p_other])

                if delta < -1e-12:
                    # Swap
                    best_mapping[lq1] = pq2
                    best_mapping[lq2] = pq1
                    best_rev[pq1] = lq2
                    best_rev[pq2] = lq1
                    best_score += delta
                    improved = True

    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_rev

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)