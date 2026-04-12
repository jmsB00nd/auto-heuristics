def init_mapping(self):
    from collections import defaultdict, deque
    import numpy as np

    num_q = self.num_qubits

    # Default trivial mapping
    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    # -------------------------------------------------------------------
    # Step 1: Build DAG and compute topological ordering
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

    # Kahn's topological sort
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

    # -------------------------------------------------------------------
    # Step 2: Identify 2-qubit gates, build time-decayed interaction matrix
    # -------------------------------------------------------------------
    two_q_gates = [g for g in schedule if len(self.access[g]) == 2]

    if not two_q_gates:
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    logical_qubits_used = set()
    for g in two_q_gates:
        logical_qubits_used.update(self.access[g])
    logical_list = sorted(logical_qubits_used)
    n_logical = len(logical_list)
    lq_to_idx = {q: i for i, q in enumerate(logical_list)}

    alpha = 2.5
    W = np.zeros((n_logical, n_logical))

    for g in two_q_gates:
        q1, q2 = self.access[g]
        i, j = lq_to_idx[q1], lq_to_idx[q2]
        r = topo_rank.get(g, 0)
        weight = np.exp(-alpha * r / total_gates)
        W[i][j] += weight
        W[j][i] += weight

    # -------------------------------------------------------------------
    # Step 3: Hardware community detection via spectral clustering
    # -------------------------------------------------------------------
    physical_nodes = sorted(self.backend.keys())
    n_physical = len(physical_nodes)
    pq_to_idx = {p: i for i, p in enumerate(physical_nodes)}

    adj_hw = np.zeros((n_physical, n_physical))
    for p in physical_nodes:
        for nb in self.backend[p]:
            if nb in pq_to_idx:
                adj_hw[pq_to_idx[p]][pq_to_idx[nb]] = 1.0

    degree_hw = np.sum(adj_hw, axis=1)
    L_hw = np.diag(degree_hw) - adj_hw
    evals_hw, evecs_hw = np.linalg.eigh(L_hw)

    # Determine k from spectral gap
    max_k = min(n_logical, n_physical, 8)
    k = 2
    if max_k > 2:
        upper = min(max_k + 2, len(evals_hw))
        gaps = np.diff(evals_hw[1:upper])
        if len(gaps) > 0:
            k = np.argmax(gaps) + 2
            k = max(2, min(k, max_k))

    # K-means++ clustering
    def kmeans(data, k, max_iter=50):
        n = data.shape[0]
        if k >= n:
            return np.arange(n)
        rng = np.random.RandomState(42)
        centers_idx = [rng.randint(n)]
        for _ in range(1, k):
            dists = np.min(
                np.array([np.sum((data - data[c]) ** 2, axis=1) for c in centers_idx]),
                axis=0,
            )
            probs = dists / (dists.sum() + 1e-12)
            centers_idx.append(rng.choice(n, p=probs))
        centroids = data[centers_idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.array([np.sum((data - c) ** 2, axis=1) for c in centroids])
            new_labels = np.argmin(dists, axis=0)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = np.mean(data[mask], axis=0)
        return labels

    # Cluster hardware qubits
    hw_features = evecs_hw[:, 1 : k + 1]
    norms = np.linalg.norm(hw_features, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    hw_features = hw_features / norms
    hw_labels = kmeans(hw_features, k)

    hw_communities = defaultdict(list)
    for idx, lbl in enumerate(hw_labels):
        hw_communities[lbl].append(physical_nodes[idx])

    # -------------------------------------------------------------------
    # Step 4: Spectral clustering of logical qubits using W
    # -------------------------------------------------------------------
    degree_W = np.sum(W, axis=1)
    L_W = np.diag(degree_W) - W
    evals_W, evecs_W = np.linalg.eigh(L_W)

    lq_features = evecs_W[:, 1 : k + 1]
    norms = np.linalg.norm(lq_features, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    lq_features = lq_features / norms
    lq_labels = kmeans(lq_features, k)

    logical_clusters = defaultdict(list)
    for idx, lbl in enumerate(lq_labels):
        logical_clusters[lbl].append(logical_list[idx])

    # -------------------------------------------------------------------
    # Step 5: Match logical clusters to hardware communities
    # -------------------------------------------------------------------
    dist_matrix = self.distance_matrix

    cluster_weight = {}
    for cid, members in logical_clusters.items():
        total_w = 0.0
        for ia, a in enumerate(members):
            for b in members[ia + 1 :]:
                total_w += W[lq_to_idx[a]][lq_to_idx[b]]
        cluster_weight[cid] = total_w

    sorted_lc = sorted(logical_clusters.keys(), key=lambda c: -cluster_weight[c])

    hw_comm_centrality = {}
    for hcid, members in hw_communities.items():
        if len(members) <= 1:
            hw_comm_centrality[hcid] = 0.0
            continue
        total_inv_dist = 0.0
        count = 0
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                d = dist_matrix[a][b]
                total_inv_dist += 1.0 / max(d, 1)
                count += 1
        hw_comm_centrality[hcid] = total_inv_dist / max(count, 1)

    used_hw = set()
    cluster_mapping = {}

    for lc in sorted_lc:
        lc_size = len(logical_clusters[lc])
        best_hc = None
        best_score = -1.0

        for hcid in hw_communities:
            if hcid in used_hw:
                continue
            if len(hw_communities[hcid]) >= lc_size:
                score = hw_comm_centrality[hcid]
                if score > best_score:
                    best_score = score
                    best_hc = hcid

        if best_hc is None:
            remaining = [h for h in hw_communities if h not in used_hw]
            if remaining:
                best_hc = max(remaining, key=lambda h: len(hw_communities[h]))
            else:
                best_hc = max(hw_communities.keys(), key=lambda h: len(hw_communities[h]))

        cluster_mapping[lc] = best_hc
        used_hw.add(best_hc)

    # -------------------------------------------------------------------
    # Step 6: Intra-cluster greedy assignment minimizing
    #         sum W[i][j] * distance(map(i), map(j))
    # -------------------------------------------------------------------
    mapping_dict = list(range(num_q))
    reverse_mapping_dict = [-1] * num_q
    occupied = set()
    assigned_set = set()

    for lc in sorted_lc:
        hcid = cluster_mapping[lc]
        log_qs = logical_clusters[lc]
        phys_pool = [p for p in hw_communities[hcid] if p not in occupied]

        # Borrow nearest unoccupied physical qubits if pool is too small
        if len(phys_pool) < len(log_qs):
            extra_needed = len(log_qs) - len(phys_pool)
            all_free = sorted(set(physical_nodes) - occupied - set(phys_pool))
            if phys_pool:
                ref = phys_pool[0]
                all_free.sort(key=lambda p: dist_matrix[ref][p])
            phys_pool.extend(all_free[:extra_needed])

        log_qs_sorted = sorted(log_qs, key=lambda q: -np.sum(W[lq_to_idx[q]]))
        available = list(phys_pool)

        for lq in log_qs_sorted:
            idx_lq = lq_to_idx[lq]
            best_pq = None
            best_cost = float("inf")

            for pq in available:
                cost = 0.0
                for prev_lq in assigned_set:
                    if prev_lq not in lq_to_idx:
                        continue
                    prev_idx = lq_to_idx[prev_lq]
                    w_val = W[idx_lq][prev_idx]
                    if w_val > 0:
                        cost += w_val * dist_matrix[pq][mapping_dict[prev_lq]]

                if cost < best_cost:
                    best_cost = cost
                    best_pq = pq

            if best_pq is None:
                best_pq = available[0]

            mapping_dict[lq] = best_pq
            reverse_mapping_dict[best_pq] = lq
            occupied.add(best_pq)
            assigned_set.add(lq)
            available.remove(best_pq)

    # -------------------------------------------------------------------
    # Step 7: Assign remaining unmapped logical qubits to free physical qubits
    # -------------------------------------------------------------------
    remaining_physical = [p for p in physical_nodes if p not in occupied]
    remaining_logical = [q for q in range(num_q) if q not in assigned_set]

    for lq, pq in zip(remaining_logical, remaining_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)