def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    from scipy.optimize import linear_sum_assignment
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # ── 0. Collect logical / physical qubits and 2-qubit gates ──────────
    logical_qubit_set = set()
    two_qubit_gates = []  # (gate_id, q1, q2)

    for gate_id, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            two_qubit_gates.append((gate_id, qubits[0], qubits[1]))

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())
    n_lq = len(logical_qubits)
    n_pq = len(physical_qubits)

    # Trivial fallback
    if n_lq <= 1 or not two_qubit_gates:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ── 1. ASAP scheduling via topological sort ─────────────────────────
    sorted_2q = sorted(two_qubit_gates, key=lambda x: x[0])
    predecessors = defaultdict(set)
    successors = defaultdict(set)
    last_gate_on_qubit = {}

    for gid, q1, q2 in sorted_2q:
        for q in (q1, q2):
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                predecessors[gid].add(pred)
                successors[pred].add(gid)
            last_gate_on_qubit[q] = gid

    in_deg = {}
    for gid, _, _ in sorted_2q:
        in_deg[gid] = len(predecessors.get(gid, set()))

    layer = {}
    queue = deque()
    for gid, _, _ in sorted_2q:
        if in_deg[gid] == 0:
            layer[gid] = 0
            queue.append(gid)

    while queue:
        g = queue.popleft()
        for s in successors.get(g, set()):
            layer[s] = max(layer.get(s, 0), layer[g] + 1)
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    # ── 2. Build time-decayed affinity matrix ───────────────────────────
    decay = 0.7
    lq_idx = {q: i for i, q in enumerate(logical_qubits)}
    affinity = np.zeros((n_lq, n_lq))

    for gid, q1, q2 in sorted_2q:
        i, j = lq_idx[q1], lq_idx[q2]
        w = decay ** layer.get(gid, 0)
        affinity[i][j] += w
        affinity[j][i] += w

    # ── Helper: flat Hungarian assignment (fallback / small circuits) ───
    def flat_hungarian():
        cost = np.zeros((n_lq, n_pq))
        mean_d = np.zeros(n_pq)
        for j, pq in enumerate(physical_qubits):
            dists = [self.distance_matrix[pq][p]
                     for p in physical_qubits
                     if p != pq and self.distance_matrix[pq][p] != float('inf')]
            mean_d[j] = sum(dists) / len(dists) if dists else 0

        for i in range(n_lq):
            wd = affinity[i].sum()
            for j in range(n_pq):
                cost[i][j] = wd * mean_d[j]

        ri, ci = linear_sum_assignment(cost)
        return {logical_qubits[r]: physical_qubits[c] for r, c in zip(ri, ci)}

    # ── 3. Determine k and cluster logical qubits ───────────────────────
    if n_lq <= 4:
        lq_to_phys = flat_hungarian()
    else:
        k = max(2, min(n_lq // 2, int(round(np.sqrt(n_lq)))))

        # Convert affinity to distance for clustering
        max_aff = affinity.max()
        if max_aff > 0:
            dist_lq = max_aff - affinity
        else:
            dist_lq = np.ones((n_lq, n_lq))
        np.fill_diagonal(dist_lq, 0)
        dist_lq = (dist_lq + dist_lq.T) / 2
        dist_lq = np.maximum(dist_lq, 0)  # numerical safety

        condensed_lq = squareform(dist_lq, checks=False)
        Z_lq = linkage(condensed_lq, method='average')
        logical_labels = fcluster(Z_lq, t=k, criterion='maxclust')

        logical_clusters = defaultdict(list)
        for i, lab in enumerate(logical_labels):
            logical_clusters[lab].append(i)
        lc_list = list(logical_clusters.values())
        k_actual = len(lc_list)

        # ── 4. Detect hardware communities ──────────────────────────────
        hw_dist = np.zeros((n_pq, n_pq))
        for i in range(n_pq):
            for j in range(n_pq):
                d = self.distance_matrix[physical_qubits[i]][physical_qubits[j]]
                hw_dist[i][j] = d if d != float('inf') else 1000

        if n_pq > k_actual and k_actual > 1:
            condensed_hw = squareform(hw_dist, checks=False)
            Z_hw = linkage(condensed_hw, method='ward')
            hw_labels = fcluster(Z_hw, t=k_actual, criterion='maxclust')
        else:
            hw_labels = np.ones(n_pq, dtype=int)

        hw_communities = defaultdict(list)
        for i, lab in enumerate(hw_labels):
            hw_communities[lab].append(i)
        hc_list = list(hw_communities.values())
        n_lc = len(lc_list)
        n_hc = len(hc_list)

        # ── 5. Match logical clusters → hardware communities (Hungarian) ─
        match_cost = np.zeros((n_lc, n_hc))

        for i, lc in enumerate(lc_list):
            total_aff = 1.0
            for a in range(len(lc)):
                for b in range(a + 1, len(lc)):
                    total_aff += affinity[lc[a]][lc[b]]

            for j, hc in enumerate(hc_list):
                if len(hc) < len(lc):
                    match_cost[i][j] = 1e9  # community too small
                    continue
                if len(hc) > 1:
                    s = sum(hw_dist[a][b]
                            for a in range(len(hc))
                            for b in range(a + 1, len(hc)))
                    cnt = len(hc) * (len(hc) - 1) // 2
                    mean_hw = s / cnt
                else:
                    mean_hw = 0
                match_cost[i][j] = total_aff * mean_hw

        # If some cluster can't fit in any community, fall back
        if np.any(np.all(match_cost >= 1e9, axis=1)):
            lq_to_phys = flat_hungarian()
        else:
            rc, cc = linear_sum_assignment(match_cost)

            # ── 6. Within each pair: fine-grained Hungarian ─────────────
            lq_to_phys = {}

            for r, c in zip(rc, cc):
                lc = lc_list[r]
                hc = hc_list[c]

                # Centrality of each physical qubit within its community
                centrality = np.zeros(len(hc))
                if len(hc) > 1:
                    for jj, hj in enumerate(hc):
                        centrality[jj] = np.mean(
                            [hw_dist[hj][hk] for hk in hc if hk != hj])

                # Inner cost: affinity-weighted centrality
                # High-affinity logical qubits → central physical qubits
                inner_cost = np.zeros((len(lc), len(hc)))
                for ii, li in enumerate(lc):
                    wd = sum(affinity[li][lj] for lj in lc if lj != li) + 1e-6
                    for jj in range(len(hc)):
                        inner_cost[ii][jj] = wd * centrality[jj]

                ri, ci = linear_sum_assignment(inner_cost)
                for rr, cc_inner in zip(ri, ci):
                    lq = logical_qubits[lc[rr]]
                    pq = physical_qubits[hc[cc_inner]]
                    lq_to_phys[lq] = pq

    # ── 7. Build strict 1-to-1 bijective mapping via in-place swaps ─────
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)