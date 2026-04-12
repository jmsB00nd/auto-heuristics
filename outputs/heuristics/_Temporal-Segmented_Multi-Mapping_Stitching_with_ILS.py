def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    import random

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)
    dist_matrix = self.distance_matrix

    # Default trivial mapping
    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    # -------------------------------------------------------------------
    # Step 1: Build DAG and compute topological layers
    # -------------------------------------------------------------------
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

    # Compute topological layers via BFS (layer = longest path from root)
    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_layer = {}
    while queue:
        g = queue.popleft()
        layer = 0
        for p in predecessors.get(g, set()):
            if p in topo_layer:
                layer = max(layer, topo_layer[p] + 1)
        topo_layer[g] = layer
        for s in successors.get(g, set()):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # Identify 2-qubit gates
    two_q_gates = [g for g in schedule if len(self.access[g]) == 2]
    if not two_q_gates:
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Logical qubits used in 2-qubit gates
    logical_qubits_set = set()
    for g in two_q_gates:
        logical_qubits_set.update(self.access[g])
    logical_list = sorted(logical_qubits_set)
    n_logical = len(logical_list)
    lq_to_idx = {q: i for i, q in enumerate(logical_list)}

    # -------------------------------------------------------------------
    # Step 2: Partition gates into 3 temporal segments
    # -------------------------------------------------------------------
    max_layer = max(topo_layer.get(g, 0) for g in two_q_gates) if two_q_gates else 0
    third = max((max_layer + 1) // 3, 1)

    segments = [[], [], []]  # early, middle, late
    for g in two_q_gates:
        layer = topo_layer.get(g, 0)
        if layer < third:
            segments[0].append(g)
        elif layer < 2 * third:
            segments[1].append(g)
        else:
            segments[2].append(g)

    # Ensure no empty segments - merge with adjacent if needed
    for i in range(3):
        if not segments[i]:
            segments[i] = list(two_q_gates)

    # -------------------------------------------------------------------
    # Step 3: Build per-segment interaction weight matrices
    # -------------------------------------------------------------------
    seg_weights = []
    for seg_gates in segments:
        W = np.zeros((n_logical, n_logical))
        for g in seg_gates:
            q1, q2 = self.access[g]
            if q1 in lq_to_idx and q2 in lq_to_idx:
                i, j = lq_to_idx[q1], lq_to_idx[q2]
                W[i][j] += 1.0
                W[j][i] += 1.0
        seg_weights.append(W)

    # Also build full interaction matrix for ILS refinement
    # Use critical-path weighting (dependency count approximated by layer depth)
    max_layer_val = max(max_layer, 1)
    W_full = np.zeros((n_logical, n_logical))
    for g in two_q_gates:
        q1, q2 = self.access[g]
        if q1 in lq_to_idx and q2 in lq_to_idx:
            i, j = lq_to_idx[q1], lq_to_idx[q2]
            layer = topo_layer.get(g, 0)
            cp_weight = 1.0 + (max_layer_val - layer) / max_layer_val
            W_full[i][j] += cp_weight
            W_full[j][i] += cp_weight

    # -------------------------------------------------------------------
    # Step 4: Multi-seed greedy placement with connectivity matching
    # -------------------------------------------------------------------
    # Precompute physical qubit degrees for connectivity matching
    phys_degree = {}
    for pq in physical_qubits:
        phys_degree[pq] = len(self.backend.get(pq, set()))

    def compute_cost(mapping, W_mat):
        """Compute sum of W[i][j] * distance(map(i), map(j)) for logical pairs."""
        cost = 0.0
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                if W_mat[i][j] > 0:
                    pi = mapping[logical_list[i]]
                    pj = mapping[logical_list[j]]
                    cost += W_mat[i][j] * dist_matrix[pi][pj]
        return cost

    def greedy_placement(W_mat, seed):
        """Multi-seed greedy: place highest-degree logical qubit first, then BFS expand."""
        rng = random.Random(seed)

        # Compute logical degree from W_mat
        log_deg = np.sum(W_mat, axis=1)
        logical_neighbors = defaultdict(dict)
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                if W_mat[i][j] > 0:
                    logical_neighbors[logical_list[i]][logical_list[j]] = W_mat[i][j]
                    logical_neighbors[logical_list[j]][logical_list[i]] = W_mat[i][j]

        mapping = [-1] * num_q
        rev_mapping = [-1] * num_q
        used_physical = set()
        placed = set()

        # Pick starting logical qubit: highest degree with some randomness
        sorted_lq = sorted(range(n_logical), key=lambda i: -log_deg[i])
        # Pick from top candidates with randomness
        top_k = max(1, min(3, n_logical))
        start_idx = sorted_lq[rng.randint(0, top_k - 1)]
        start_lq = logical_list[start_idx]

        # Pick starting physical qubit: match degree, with randomness
        target_deg = log_deg[start_idx]
        phys_sorted = sorted(physical_qubits, key=lambda p: abs(phys_degree[p] - target_deg))
        top_phys = max(1, min(3, n_phys))
        start_pq = phys_sorted[rng.randint(0, top_phys - 1)]

        mapping[start_lq] = start_pq
        rev_mapping[start_pq] = start_lq
        used_physical.add(start_pq)
        placed.add(start_lq)

        # BFS-like greedy expansion
        bfs_queue = deque([start_lq])
        while len(placed) < n_logical:
            if bfs_queue:
                current_lq = bfs_queue.popleft()
            else:
                # Pick unplaced qubit with highest interaction to placed set
                remaining = [lq for lq in logical_list if lq not in placed]
                best_lq = max(remaining, key=lambda lq: sum(
                    logical_neighbors.get(lq, {}).get(plq, 0.0) for plq in placed
                ))
                current_lq = best_lq

            # Find unplaced neighbors of current_lq, sorted by interaction weight
            unplaced_neighbors = []
            for nb_lq, w in sorted(logical_neighbors.get(current_lq, {}).items(),
                                     key=lambda x: -x[1]):
                if nb_lq not in placed:
                    unplaced_neighbors.append(nb_lq)

            for nb_lq in unplaced_neighbors:
                if nb_lq in placed:
                    continue

                # Find best physical qubit near placed neighbors
                current_pq = mapping[current_lq]
                # Candidates: neighbors of physical qubits of placed logical neighbors
                candidates = set()
                for plq in placed:
                    if plq in logical_neighbors.get(nb_lq, {}):
                        ppq = mapping[plq]
                        for adj in self.backend.get(ppq, set()):
                            if adj not in used_physical:
                                candidates.add(adj)

                if not candidates:
                    # Expand to 2-hop neighbors
                    for adj in self.backend.get(current_pq, set()):
                        if adj not in used_physical:
                            candidates.add(adj)
                        for adj2 in self.backend.get(adj, set()):
                            if adj2 not in used_physical:
                                candidates.add(adj2)

                if not candidates:
                    # Fallback: all free physical qubits
                    candidates = set(pq for pq in physical_qubits if pq not in used_physical)

                if not candidates:
                    break

                # Score candidates by weighted distance to placed neighbors
                nb_idx = lq_to_idx[nb_lq]
                best_pq = None
                best_cost = float('inf')
                for cpq in candidates:
                    cost = 0.0
                    for plq in placed:
                        pidx = lq_to_idx[plq]
                        if W_mat[nb_idx][pidx] > 0:
                            cost += W_mat[nb_idx][pidx] * dist_matrix[cpq][mapping[plq]]
                    if cost < best_cost:
                        best_cost = cost
                        best_pq = cpq

                mapping[nb_lq] = best_pq
                rev_mapping[best_pq] = nb_lq
                used_physical.add(best_pq)
                placed.add(nb_lq)
                bfs_queue.append(nb_lq)

            # If current_lq had no unplaced neighbors, we continue BFS
            if current_lq not in placed:
                placed.add(current_lq)

        # Fill remaining unmapped logical qubits
        unmapped_logical = [q for q in range(num_q) if mapping[q] == -1]
        free_physical = [pq for pq in range(num_q) if rev_mapping[pq] == -1]
        for lq, pq in zip(unmapped_logical, free_physical):
            mapping[lq] = pq
            rev_mapping[pq] = lq

        return mapping, rev_mapping

    # -------------------------------------------------------------------
    # Step 5: Generate M=5 candidate mappings per segment
    # -------------------------------------------------------------------
    M = 5
    T = 3
    segment_candidates = []  # segment_candidates[t] = list of (mapping, rev_mapping, cost)

    for t in range(T):
        candidates = []
        for seed in range(M):
            m, r = greedy_placement(seg_weights[t], seed * 17 + t * 101 + 42)
            cost = compute_cost(m, seg_weights[t])
            candidates.append((m, r, cost))
        segment_candidates.append(candidates)

    # -------------------------------------------------------------------
    # Step 6: Evaluate all 5^3 = 125 triples, pick best segment-1 mapping
    # -------------------------------------------------------------------
    alpha_w, beta_w, gamma_w, lambda_w = 3.0, 2.0, 1.0, 0.5

    def transition_cost(map_a, map_b):
        """Approximate token-swap transition cost: sum of distances of displaced qubits."""
        cost = 0
        for lq in logical_list:
            pa = map_a[lq]
            pb = map_b[lq]
            if pa != pb:
                cost += dist_matrix[pa][pb]
        return cost

    best_total_score = float('inf')
    best_seg1_idx = 0

    for i in range(M):
        ci = segment_candidates[0][i]
        cost_i = ci[2]
        for j in range(M):
            cj = segment_candidates[1][j]
            cost_j = cj[2]
            tc_ij = transition_cost(ci[0], cj[0])
            for k in range(M):
                ck = segment_candidates[2][k]
                cost_k = ck[2]
                tc_jk = transition_cost(cj[0], ck[0])

                total = (alpha_w * cost_i + beta_w * cost_j + gamma_w * cost_k +
                         lambda_w * (tc_ij + tc_jk))

                if total < best_total_score:
                    best_total_score = total
                    best_seg1_idx = i

    # Select the segment-1 mapping from the best triple
    best_mapping = list(segment_candidates[0][best_seg1_idx][0])
    best_rev_mapping = list(segment_candidates[0][best_seg1_idx][1])

    # -------------------------------------------------------------------
    # Step 7: ILS + SA refinement using full critical-path-weighted cost
    # -------------------------------------------------------------------
    def full_cost(mapping):
        """Cost using full critical-path weighted interaction matrix."""
        cost = 0.0
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                if W_full[i][j] > 0:
                    pi = mapping[logical_list[i]]
                    pj = mapping[logical_list[j]]
                    cost += W_full[i][j] * dist_matrix[pi][pj]
        return cost

    def swap_in_mapping(mapping, rev_mapping, pq1, pq2):
        """Swap two physical qubits in the mapping."""
        lq1 = rev_mapping[pq1]
        lq2 = rev_mapping[pq2]
        mapping[lq1], mapping[lq2] = pq2, pq1
        rev_mapping[pq1], rev_mapping[pq2] = lq2, lq1

    current_mapping = best_mapping
    current_rev = best_rev_mapping
    current_cost = full_cost(current_mapping)

    best_ever_mapping = list(current_mapping)
    best_ever_rev = list(current_rev)
    best_ever_cost = current_cost

    rng = random.Random(12345)
    T_sa = 2.0  # Initial SA temperature
    T_min = 0.01
    cooling = 0.97
    max_ils_iters = min(150, n_logical * 10)
    no_improve_limit = max(30, n_logical * 2)
    no_improve_count = 0

    # Get physical qubits that have logical qubits mapped to them
    active_physical = [current_mapping[lq] for lq in logical_list]

    for iteration in range(max_ils_iters):
        if no_improve_count >= no_improve_limit:
            break

        # Local search: try random adjacent swap
        pq1 = rng.choice(active_physical)
        neighbors = list(self.backend.get(pq1, set()))
        if not neighbors:
            continue
        pq2 = rng.choice(neighbors)

        # Try swap
        swap_in_mapping(current_mapping, current_rev, pq1, pq2)
        new_cost = full_cost(current_mapping)

        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / max(T_sa, 1e-10)):
            current_cost = new_cost
            if current_cost < best_ever_cost:
                best_ever_cost = current_cost
                best_ever_mapping = list(current_mapping)
                best_ever_rev = list(current_rev)
                no_improve_count = 0
                # Update active physical qubits
                active_physical = [current_mapping[lq] for lq in logical_list]
            else:
                no_improve_count += 1
        else:
            # Revert swap
            swap_in_mapping(current_mapping, current_rev, pq1, pq2)
            no_improve_count += 1

        T_sa *= cooling

        # ILS perturbation: every 20 iterations with no improvement, do a random double-swap
        if no_improve_count > 0 and no_improve_count % 20 == 0:
            # Perturbation: random swap of two logical qubits' physical assignments
            if len(logical_list) >= 2:
                lq_a, lq_b = rng.sample(logical_list, 2)
                pa, pb = current_mapping[lq_a], current_mapping[lq_b]
                swap_in_mapping(current_mapping, current_rev, pa, pb)
                current_cost = full_cost(current_mapping)
                active_physical = [current_mapping[lq] for lq in logical_list]
                # Reset SA temperature partially
                T_sa = max(T_sa, 0.5)

    self.mapping_dict = best_ever_mapping
    self.reverse_mapping_dict = best_ever_rev

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)