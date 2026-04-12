def init_mapping(self):
    import math
    import random
    from collections import defaultdict, deque
    from time import time as _time

    random.seed(42)
    num_q = self.num_qubits
    dist = self.distance_matrix
    backend = self.backend
    physical_qubits = sorted(backend.keys())
    n_phys = len(physical_qubits)

    # ---------------------------------------------------------------
    # Step 1: Build DAG, topological order, critical-path weights
    # ---------------------------------------------------------------
    all_gates = sorted(self.access.keys())
    gates_2q = {}
    logical_qubits_set = set()
    last_gate_on_qubit = {}
    succ_dag = defaultdict(set)
    pred_dag = defaultdict(set)

    for g in all_gates:
        qubits = self.access[g]
        for q in qubits:
            if q in last_gate_on_qubit:
                succ_dag[last_gate_on_qubit[q]].add(g)
                pred_dag[g].add(last_gate_on_qubit[q])
            last_gate_on_qubit[q] = g
        if len(qubits) == 2:
            gates_2q[g] = (qubits[0], qubits[1])
            logical_qubits_set.add(qubits[0])
            logical_qubits_set.add(qubits[1])
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)

    if not gates_2q:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # 2q-only DAG
    dag2q_succ = defaultdict(set)
    dag2q_pred = defaultdict(set)
    last_2q_on_qubit = {}
    for g in all_gates:
        qubits = self.access[g]
        if len(qubits) == 2:
            for q in qubits:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != g:
                        dag2q_succ[prev].add(g)
                        dag2q_pred[g].add(prev)
                last_2q_on_qubit[q] = g

    # Topological sort and critical-path (descendant depth)
    in_deg_2q = {g: len(dag2q_pred[g]) for g in gates_2q}
    topo_2q = []
    q_topo = deque(g for g in gates_2q if in_deg_2q[g] == 0)
    while q_topo:
        g = q_topo.popleft()
        topo_2q.append(g)
        for s in dag2q_succ[g]:
            in_deg_2q[s] -= 1
            if in_deg_2q[s] == 0:
                q_topo.append(s)

    dep_count = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count[g] = max(dep_count[g], dep_count[s] + 1)

    # Gate layers
    gate_layer = {g: 0 for g in gates_2q}
    for g in topo_2q:
        for s in dag2q_succ[g]:
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
    max_layer = max(gate_layer.values(), default=1) or 1

    # ---------------------------------------------------------------
    # Step 2: Interaction weights with critical-path weighting
    # ---------------------------------------------------------------
    static_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for g, (q1, q2) in gates_2q.items():
        cp = dep_count[g] + 1
        layer = gate_layer[g]
        w = cp * (max_layer - layer + 1)
        key = (min(q1, q2), max(q1, q2))
        static_weight[key] += w
        logical_degree[q1] += w
        logical_degree[q2] += w

    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    # ---------------------------------------------------------------
    # Step 3: Physical graph properties
    # ---------------------------------------------------------------
    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in backend.get(pq, []):
            hw_adj[pq].add(pq2)

    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(dist[pq][pq2] for pq2 in physical_qubits)

    # ---------------------------------------------------------------
    # Step 4: Helper functions
    # ---------------------------------------------------------------
    def build_neighbors(weights):
        nbrs = defaultdict(dict)
        deg = defaultdict(float)
        for (q1, q2), w in weights.items():
            nbrs[q1][q2] = w
            nbrs[q2][q1] = w
            deg[q1] += w
            deg[q2] += w
        return nbrs, deg

    def compute_cost(m, weights):
        c = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c += w * dist[m[q1]][m[q2]]
        return c

    def compute_cost_nonlinear(m, weights, alpha_exp):
        c = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c += w * (dist[m[q1]][m[q2]] ** alpha_exp)
        return c

    def delta_swap_cost_nonlinear(m, rm, pq_a, pq_b, nbrs, alpha_exp):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        delta = 0.0
        affected = set()
        if lq_a in nbrs:
            affected.update(nbrs[lq_a].keys())
        if lq_b in nbrs:
            affected.update(nbrs[lq_b].keys())
        for qq in affected:
            if qq == lq_a or qq == lq_b:
                continue
            pq_q = m[qq]
            if pq_q < 0:
                continue
            w_a = nbrs.get(lq_a, {}).get(qq, 0.0)
            if w_a > 0:
                delta += w_a * (dist[pq_b][pq_q] ** alpha_exp - dist[pq_a][pq_q] ** alpha_exp)
            w_b = nbrs.get(lq_b, {}).get(qq, 0.0)
            if w_b > 0:
                delta += w_b * (dist[pq_a][pq_q] ** alpha_exp - dist[pq_b][pq_q] ** alpha_exp)
        return delta

    def do_swap(m, rm, pq_a, pq_b):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        if lq_a >= 0:
            m[lq_a] = pq_b
        if lq_b >= 0:
            m[lq_b] = pq_a
        rm[pq_a], rm[pq_b] = lq_b, lq_a

    def fill_unmapped(m, rm):
        unmapped = [q for q in range(num_q) if m[q] == -1]
        free = [pq for pq in range(num_q) if rm[pq] == -1]
        for lq, pq in zip(unmapped, free):
            m[lq] = pq
            rm[pq] = lq

    # ---------------------------------------------------------------
    # Step 5: Seed generation — MIS-based
    # ---------------------------------------------------------------
    static_nbrs, static_deg = build_neighbors(static_weight)
    max_iw = max(static_weight.values()) if static_weight else 1.0

    def mis_seed(start_pq):
        if len(interacting_logical) < 2:
            return None, None
        # Top-3 partner conflict graph
        top_partners = {}
        for lq in interacting_logical:
            nbr_list = sorted(static_nbrs.get(lq, {}).items(), key=lambda x: x[1], reverse=True)
            top_partners[lq] = set(p for p, _ in nbr_list[:3])

        conflict_adj = defaultdict(set)
        il = interacting_logical
        for i in range(len(il)):
            for j in range(i + 1, len(il)):
                s1 = top_partners.get(il[i], set())
                s2 = top_partners.get(il[j], set())
                union = len(s1 | s2)
                if union > 0 and len(s1 & s2) / union > 0.5:
                    conflict_adj[il[i]].add(il[j])
                    conflict_adj[il[j]].add(il[i])

        sorted_by_deg = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
        mis = set()
        excluded = set()
        for lq in sorted_by_deg:
            if lq not in excluded:
                mis.add(lq)
                for nb in conflict_adj.get(lq, set()):
                    excluded.add(nb)

        non_mis = [lq for lq in sorted_by_deg if lq not in mis]
        mis_ordered = sorted(mis, key=lambda q: logical_degree[q], reverse=True)

        m = [-1] * num_q
        rm = [-1] * num_q
        used_phys = set()

        if mis_ordered:
            m[mis_ordered[0]] = start_pq
            rm[start_pq] = mis_ordered[0]
            used_phys.add(start_pq)

            for lq in mis_ordered[1:]:
                nbrs_placed = {plq: static_nbrs[lq][plq] for plq in mis if m[plq] >= 0 and plq in static_nbrs.get(lq, {})}
                if nbrs_placed:
                    best_pq, best_s = None, float('inf')
                    for pq in physical_qubits:
                        if pq in used_phys:
                            continue
                        s = sum(iw * dist[pq][m[plq]] * (0.9 - 0.1 * iw / max_iw if m[plq] in hw_adj[pq] else 1.0) for plq, iw in nbrs_placed.items())
                        if s < best_s:
                            best_s, best_pq = s, pq
                else:
                    best_pq = min((pq for pq in physical_qubits if pq not in used_phys), key=lambda pq: phys_centrality[pq], default=None)

                if best_pq is not None:
                    m[lq] = best_pq
                    rm[best_pq] = lq
                    used_phys.add(best_pq)

        placed = set(lq for lq in interacting_logical if m[lq] >= 0)
        for lq in non_mis:
            nbrs_placed = {plq: static_nbrs.get(lq, {}).get(plq, 0.0) for plq in placed if plq in static_nbrs.get(lq, {})}
            if nbrs_placed:
                best_pq, best_s = None, float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    s = sum(iw * dist[pq][m[plq]] * (0.9 - 0.1 * iw / max_iw if m[plq] in hw_adj[pq] else 1.0) for plq, iw in nbrs_placed.items())
                    if s < best_s:
                        best_s, best_pq = s, pq
            else:
                best_pq = min((pq for pq in physical_qubits if pq not in used_phys), key=lambda pq: phys_centrality[pq], default=None)
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                used_phys.add(best_pq)
                placed.add(lq)

        # Remaining logical qubits
        for lq in logical_qubits:
            if m[lq] == -1:
                best_pq = min((pq for pq in physical_qubits if pq not in used_phys), key=lambda pq: phys_centrality[pq], default=None)
                if best_pq is not None:
                    m[lq] = best_pq
                    rm[best_pq] = lq
                    used_phys.add(best_pq)
        return m, rm

    # ---------------------------------------------------------------
    # Step 6: Greedy seed
    # ---------------------------------------------------------------
    def greedy_seed(start_lq, start_pq):
        m = [-1] * num_q
        rm = [-1] * num_q
        m[start_lq] = start_pq
        rm[start_pq] = start_lq
        used_phys = {start_pq}
        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            best_lq, best_w = None, -1.0
            for lq in remaining:
                w = sum(static_nbrs.get(lq, {}).get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w, best_lq = w, lq

            nbrs_placed = {plq: static_nbrs.get(best_lq, {}).get(plq, 0.0) for plq in placed if plq in static_nbrs.get(best_lq, {})}
            if nbrs_placed:
                best_pq, best_s = None, float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    s = sum(iw * dist[pq][m[plq]] * (0.9 - 0.1 * iw / max_iw if m[plq] in hw_adj[pq] else 1.0) for plq, iw in nbrs_placed.items())
                    if s < best_s:
                        best_s, best_pq = s, pq
            else:
                best_pq = min((pq for pq in physical_qubits if pq not in used_phys), key=lambda pq: phys_centrality[pq], default=None)

            if best_pq is not None:
                m[best_lq] = best_pq
                rm[best_pq] = best_lq
                used_phys.add(best_pq)
            placed.add(best_lq)
            remaining.discard(best_lq)
        return m, rm

    # ---------------------------------------------------------------
    # Step 7: Spectral seed
    # ---------------------------------------------------------------
    def spectral_seed():
        if len(interacting_logical) < 3:
            return None, None

        idx_map = {q: i for i, q in enumerate(interacting_logical)}
        n = len(interacting_logical)
        L = [[0.0] * n for _ in range(n)]
        for (q1, q2), w in static_weight.items():
            if q1 in idx_map and q2 in idx_map:
                i, j = idx_map[q1], idx_map[q2]
                L[i][j] -= w
                L[j][i] -= w
                L[i][i] += w
                L[j][j] += w

        def power_iter(mat, n_iter=200):
            v = [random.gauss(0, 1) for _ in range(n)]
            for _ in range(n_iter):
                new_v = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
                norm = math.sqrt(sum(x * x for x in new_v)) or 1e-12
                v = [x / norm for x in new_v]
            return v

        max_diag = max(L[i][i] for i in range(n)) + 1.0
        shifted = [[-L[i][j] if i != j else max_diag - L[i][j] for j in range(n)] for i in range(n)]
        v1 = power_iter(shifted, 150)
        dot_v1 = sum(x * x for x in v1) or 1e-12
        lam1 = sum(v1[i] * sum(shifted[i][j] * v1[j] for j in range(n)) for i in range(n)) / dot_v1
        shifted2 = [row[:] for row in shifted]
        for i in range(n):
            for j in range(n):
                shifted2[i][j] -= lam1 * v1[i] * v1[j] / dot_v1
        fiedler = power_iter(shifted2, 200)

        fiedler_order = sorted(range(n), key=lambda i: fiedler[i])
        sorted_logical = [interacting_logical[i] for i in fiedler_order]

        start_pq = min(physical_qubits, key=lambda pq: phys_centrality[pq])
        visited = set()
        bfs_order = []
        bfs_q = deque([start_pq])
        visited.add(start_pq)
        while bfs_q:
            pq = bfs_q.popleft()
            bfs_order.append(pq)
            for nb in sorted(hw_adj[pq]):
                if nb not in visited:
                    visited.add(nb)
                    bfs_q.append(nb)
        for pq in physical_qubits:
            if pq not in visited:
                bfs_order.append(pq)
                visited.add(pq)

        m = [-1] * num_q
        rm = [-1] * num_q
        n_interact = len(sorted_logical)
        offset = max(0, (len(bfs_order) - n_interact) // 2)
        for i, lq in enumerate(sorted_logical):
            pq = bfs_order[min(offset + i, len(bfs_order) - 1)]
            if rm[pq] != -1:
                best_pq = min((cpq for cpq in physical_qubits if rm[cpq] == -1), key=lambda cpq: dist[pq][cpq], default=None)
                pq = best_pq if best_pq is not None else pq
            if pq is not None and rm[pq] == -1:
                m[lq] = pq
                rm[pq] = lq
        return m, rm

    # ---------------------------------------------------------------
    # Step 8: Local search with nonlinear distance penalty
    # ---------------------------------------------------------------
    def local_search(m, rm, nbrs, weights, alpha_exp, max_rounds=5):
        if len(interacting_logical) <= 1:
            return compute_cost_nonlinear(m, weights, alpha_exp)
        for _ in range(max_rounds):
            best_d, best_pair = 0.0, None
            active_phys = [m[q] for q in interacting_logical if m[q] >= 0]
            for pq1 in active_phys:
                for pq2 in hw_adj.get(pq1, set()):
                    d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                    if d < best_d:
                        best_d, best_pair = d, (pq1, pq2)
            n_random = min(150, len(interacting_logical) * 3)
            for _ in range(n_random):
                i, j = random.sample(range(len(interacting_logical)), 2)
                pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                if pq1 >= 0 and pq2 >= 0:
                    d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                    if d < best_d:
                        best_d, best_pair = d, (pq1, pq2)
            if best_pair and best_d < -1e-12:
                do_swap(m, rm, *best_pair)
            else:
                break
        return compute_cost_nonlinear(m, weights, alpha_exp)

    # ---------------------------------------------------------------
    # Step 9: Routing simulation for RSDIWR (returns per-qubit swap counts)
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=20):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_involvement = defaultdict(float)  # logical qubit -> swap count

        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set(g for g in gates_2q if pred_remaining[g] == 0)
        layers_done = 0

        while front and layers_done < max_layers:
            executable = []
            for g in front:
                gq1, gq2 = gates_2q[g]
                p1, p2 = sim_m[gq1], sim_m[gq2]
                if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                    executable.append(g)

            if executable:
                for g in executable:
                    front.discard(g)
                    for s in dag2q_succ[g]:
                        pred_remaining[s] -= 1
                        if pred_remaining[s] == 0:
                            front.add(s)
                layers_done += 1
                continue

            active_phys = set()
            for g in front:
                gq1, gq2 = gates_2q[g]
                active_phys.add(sim_m[gq1])
                active_phys.add(sim_m[gq2])

            candidates_sw = set()
            for pq in active_phys:
                for nb in backend.get(pq, []):
                    candidates_sw.add((min(pq, nb), max(pq, nb)))

            best_swap, best_score = None, float('inf')
            for (s1, s2) in candidates_sw:
                l1, l2 = sim_rm[s1], sim_rm[s2]
                score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1, p2 = sim_m[gq1], sim_m[gq2]
                    if gq1 == l1: p1 = s2
                    elif gq1 == l2: p1 = s1
                    if gq2 == l1: p2 = s2
                    elif gq2 == l2: p2 = s1
                    deps = dep_count.get(g, 0) + 1
                    score += deps * dist[p1][p2]
                if score < best_score:
                    best_score, best_swap = score, (s1, s2)

            if best_swap is None:
                break

            s1, s2 = best_swap
            l1, l2 = sim_rm[s1], sim_rm[s2]

            # Track per-logical-qubit swap involvement
            if l1 >= 0 and l1 in logical_qubits_set:
                swap_involvement[l1] += 1.0
            if l2 >= 0 and l2 in logical_qubits_set:
                swap_involvement[l2] += 1.0

            sim_m[l1], sim_m[l2] = s2, s1
            sim_rm[s1], sim_rm[s2] = l2, l1

        # Also return pair-based swap counts for weight blending
        return swap_involvement

    # ---------------------------------------------------------------
    # Step 10: Exact Sub-QAP solver (branch-and-bound)
    # ---------------------------------------------------------------
    def solve_sub_qap_exact(hotspot_lqs, m, rm, weights):
        """
        Solve exact sub-QAP for hotspot logical qubits.
        hotspot_lqs: list of logical qubit IDs (the K hotspot qubits)
        Returns: new (m, rm) with hotspot qubits optimally reassigned, or None if no improvement.
        """
        K = len(hotspot_lqs)
        if K < 2:
            return None

        # Current physical positions of hotspot qubits
        hotspot_phys = [m[lq] for lq in hotspot_lqs]

        # Expand candidate physical positions: hotspot positions + free adjacent
        candidate_set = set(hotspot_phys)
        all_used = set(m[lq] for lq in range(num_q) if m[lq] >= 0)
        hotspot_phys_set = set(hotspot_phys)

        for hp in hotspot_phys:
            for nb in hw_adj.get(hp, set()):
                if nb not in all_used or nb in hotspot_phys_set:
                    candidate_set.add(nb)

        candidate_list = list(candidate_set)
        # Limit candidate set size for tractability
        if len(candidate_list) > K + 6:
            extras = [p for p in candidate_list if p not in hotspot_phys_set]
            extras.sort(key=lambda p: min(dist[p][hp] for hp in hotspot_phys))
            candidate_list = list(hotspot_phys) + extras[:6]

        M = len(candidate_list)
        if M < K:
            return None

        # Build sub-QAP matrices
        # W_sub[i][j] = interaction weight between hotspot_lqs[i] and hotspot_lqs[j]
        W_sub = [[0.0] * K for _ in range(K)]
        for i in range(K):
            for j in range(K):
                if i != j:
                    key = (min(hotspot_lqs[i], hotspot_lqs[j]), max(hotspot_lqs[i], hotspot_lqs[j]))
                    W_sub[i][j] = weights.get(key, 0.0)

        # Also include cross-terms with non-hotspot placed qubits (as fixed penalty)
        # This ensures we account for interactions outside the hotspot
        non_hotspot_placed = [(lq, m[lq]) for lq in interacting_logical if lq not in set(hotspot_lqs) and m[lq] >= 0]

        # D_sub[a][b] = distance between candidate_list[a] and candidate_list[b]
        D_sub = [[dist[candidate_list[a]][candidate_list[b]] for b in range(M)] for a in range(M)]

        # Pre-compute cross-term costs for each (hotspot_idx, candidate_position)
        cross_cost = [[0.0] * M for _ in range(K)]
        for i in range(K):
            lq_i = hotspot_lqs[i]
            for plq, ppq in non_hotspot_placed:
                key = (min(lq_i, plq), max(lq_i, plq))
                w = weights.get(key, 0.0)
                if w > 0:
                    for ci in range(M):
                        cross_cost[i][ci] += w * dist[candidate_list[ci]][ppq]

        best_cost = [float('inf')]
        best_perm = [None]
        max_nodes = [0]
        NODE_LIMIT = 500000

        def bb(partial, used, depth, cost_so_far):
            if max_nodes[0] > NODE_LIMIT:
                return
            max_nodes[0] += 1

            if depth == K:
                if cost_so_far < best_cost[0]:
                    best_cost[0] = cost_so_far
                    best_perm[0] = partial[:]
                return

            if cost_so_far >= best_cost[0]:
                return

            # Sort candidates by incremental cost for better pruning
            cands = []
            for p in range(M):
                if p in used:
                    continue
                inc = cross_cost[depth][p]
                for i in range(depth):
                    inc += W_sub[i][depth] * D_sub[partial[i]][p]
                    inc += W_sub[depth][i] * D_sub[p][partial[i]]
                cands.append((inc, p))
            cands.sort()

            for inc, p in cands:
                nc = cost_so_far + inc
                if nc >= best_cost[0]:
                    break
                partial[depth] = p
                used.add(p)
                bb(partial, used, depth + 1, nc)
                used.discard(p)

        # Compute current cost as initial upper bound
        cur_cost = 0.0
        for i in range(K):
            cur_cost += cross_cost[i][candidate_list.index(hotspot_phys[i])]
            for j in range(i + 1, K):
                key = (min(hotspot_lqs[i], hotspot_lqs[j]), max(hotspot_lqs[i], hotspot_lqs[j]))
                w = weights.get(key, 0.0)
                cur_cost += w * dist[hotspot_phys[i]][hotspot_phys[j]]
                cur_cost += w * dist[hotspot_phys[j]][hotspot_phys[i]]
        best_cost[0] = cur_cost + 1e-6  # Allow strictly improving solutions

        bb([-1] * K, set(), 0, 0.0)

        if best_perm[0] is None:
            return None

        # Apply sub-assignment atomically
        new_m = list(m)
        new_rm = list(rm)

        # Free old positions
        for lq in hotspot_lqs:
            old_pq = new_m[lq]
            new_rm[old_pq] = -1
            new_m[lq] = -1

        # Assign new positions
        for i in range(K):
            pq = candidate_list[best_perm[0][i]]
            new_m[hotspot_lqs[i]] = pq
            new_rm[pq] = hotspot_lqs[i]

        # Verify bijectivity
        assigned_phys = [new_m[lq] for lq in range(num_q) if new_m[lq] >= 0]
        if len(assigned_phys) != len(set(assigned_phys)):
            return None  # Collision detected, abort

        return new_m, new_rm

    # ---------------------------------------------------------------
    # Step 11: Perturbation modes with bandit selection
    # ---------------------------------------------------------------
    def perturb_random(m, rm, **kw):
        if len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    def perturb_segment(m, rm, **kw):
        k = min(random.randint(3, 5), len(interacting_logical))
        if k < 2:
            perturb_random(m, rm)
            return
        lqs = random.sample(interacting_logical, k)
        phys_positions = [m[lq] for lq in lqs]
        random.shuffle(phys_positions)
        for lq in lqs:
            rm[m[lq]] = -1
        for lq, pq in zip(lqs, phys_positions):
            m[lq] = pq
            rm[pq] = lq

    def perturb_worst(m, rm, weights=None, alpha_exp=1.5, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        pair_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                pair_costs.append((w * (dist[m[q1]][m[q2]] ** alpha_exp), q1, q2))
        if not pair_costs:
            return
        pair_costs.sort(reverse=True)
        _, tq1, tq2 = pair_costs[random.randint(0, min(2, len(pair_costs) - 1))]
        adj_list = list(hw_adj.get(m[tq2], set()))
        if adj_list:
            target = random.choice(adj_list)
            do_swap(m, rm, m[tq1], target)
        else:
            do_swap(m, rm, m[tq1], m[tq2])

    def perturb_lns(m, rm, nbrs=None, alpha_exp=1.5, **kw):
        if nbrs is None or len(interacting_logical) < 3:
            perturb_random(m, rm)
            return
        k = min(random.randint(3, 8), len(interacting_logical))
        qcost = {}
        for lq in interacting_logical:
            c = sum(w * (dist[m[lq]][m[p]] ** alpha_exp) for p, w in nbrs.get(lq, {}).items() if m[p] >= 0 and m[lq] >= 0)
            qcost[lq] = c
        sorted_qs = sorted(qcost, key=lambda q: qcost[q], reverse=True)
        top_half = sorted_qs[:max(k, len(sorted_qs) // 2)]
        subset = random.sample(top_half, min(k, len(top_half)))

        freed = []
        placed_set = set(lq for lq in interacting_logical if m[lq] >= 0 and lq not in subset)
        for lq in subset:
            freed.append(m[lq])
            rm[m[lq]] = -1
            m[lq] = -1

        for lq in sorted(subset, key=lambda q: qcost[q], reverse=True):
            best_pq, best_sc = None, float('inf')
            for pq in freed:
                if rm[pq] != -1:
                    continue
                sc = sum(w * (dist[pq][m[p]] ** alpha_exp) for p, w in nbrs.get(lq, {}).items() if p in placed_set and m[p] >= 0)
                if sc < best_sc:
                    best_sc, best_pq = sc, pq
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                placed_set.add(lq)
            else:
                for pq in freed:
                    if rm[pq] == -1:
                        m[lq] = pq
                        rm[pq] = lq
                        placed_set.add(lq)
                        break

    perturbation_modes = [perturb_random, perturb_segment, perturb_worst, perturb_lns]
    N_MODES = len(perturbation_modes)
    mode_successes = [0] * N_MODES
    mode_attempts = [0] * N_MODES
    EPSILON = 0.1

    def select_mode():
        if random.random() < EPSILON or sum(mode_attempts) < N_MODES * 2:
            return random.randint(0, N_MODES - 1)
        rates = [mode_successes[k] / mode_attempts[k] if mode_attempts[k] > 0 else 1.0 for k in range(N_MODES)]
        total = sum(rates)
        if total < 1e-12:
            return random.randint(0, N_MODES - 1)
        r = random.random() * total
        cumul = 0.0
        for k in range(N_MODES):
            cumul += rates[k]
            if r <= cumul:
                return k
        return N_MODES - 1

    # ---------------------------------------------------------------
    # Step 12: Generate seed candidates
    # ---------------------------------------------------------------
    sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    candidates = []

    # Greedy seeds
    for s_lq in seed_lqs:
        for s_pq in seed_pqs:
            m_c, rm_c = greedy_seed(s_lq, s_pq)
            fill_unmapped(m_c, rm_c)
            candidates.append((compute_cost(m_c, static_weight), m_c, rm_c))

    # MIS seeds
    for s_pq in seed_pqs:
        m_c, rm_c = mis_seed(s_pq)
        if m_c is not None:
            fill_unmapped(m_c, rm_c)
            candidates.append((compute_cost(m_c, static_weight), m_c, rm_c))

    # Spectral seed
    m_s, rm_s = spectral_seed()
    if m_s is not None:
        fill_unmapped(m_s, rm_s)
        candidates.append((compute_cost(m_s, static_weight), m_s, rm_s))

    # Random seed
    m_rand = [-1] * num_q
    rm_rand = [-1] * num_q
    shuffled = list(physical_qubits)
    random.shuffle(shuffled)
    for i, lq in enumerate(logical_qubits):
        if i < len(shuffled):
            m_rand[lq] = shuffled[i]
            rm_rand[shuffled[i]] = lq
    fill_unmapped(m_rand, rm_rand)
    candidates.append((compute_cost(m_rand, static_weight), m_rand, rm_rand))

    if not candidates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    candidates.sort(key=lambda x: x[0])

    # Refine top candidates
    _, best_m, best_rm = candidates[0]
    best_m, best_rm = list(best_m), list(best_rm)
    local_search(best_m, best_rm, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=6)
    best_cost = compute_cost(best_m, static_weight)

    for idx in range(1, min(5, len(candidates))):
        _, m_c, rm_c = candidates[idx]
        m_c, rm_c = list(m_c), list(rm_c)
        local_search(m_c, rm_c, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=4)
        c = compute_cost(m_c, static_weight)
        if c < best_cost:
            best_cost = c
            best_m, best_rm = list(m_c), list(rm_c)

    # ---------------------------------------------------------------
    # Step 13: Dual-Phase ILS with RSDIWR + Sub-QAP Rotation
    # ---------------------------------------------------------------
    time_budget = 25.0
    t_start = _time()
    n_rsdiwr = 4

    cur_m = list(best_m)
    cur_rm = list(best_rm)
    swap_involvement = defaultdict(float)

    for rsdiwr_iter in range(n_rsdiwr):
        elapsed = _time() - t_start
        if elapsed > time_budget:
            break

        remaining_time = time_budget - elapsed
        remaining_iters = n_rsdiwr - rsdiwr_iter
        time_for_this = remaining_time / max(remaining_iters, 1)
        n_ils = max(55, int(num_q * time_for_this / 0.5)) if time_for_this > 1.0 else max(30, num_q // 3)

        # Blend static weights with RSDIWR feedback
        if rsdiwr_iter == 0 or not swap_involvement:
            eff_weights = dict(static_weight)
        else:
            # Convert per-qubit involvement to pair weights
            pair_feedback = defaultdict(float)
            for (q1, q2) in static_weight:
                inv = swap_involvement.get(q1, 0.0) + swap_involvement.get(q2, 0.0)
                pair_feedback[(q1, q2)] = inv

            max_sw = max(pair_feedback.values()) if pair_feedback else 1.0
            scale = max(static_weight.values()) / max(max_sw, 1e-10)
            alpha_blend = max(0.3, 1.0 - 0.3 * rsdiwr_iter)
            eff_weights = defaultdict(float)
            all_keys = set(static_weight.keys()) | set(pair_feedback.keys())
            for key in all_keys:
                w_s = static_weight.get(key, 0.0)
                w_r = pair_feedback.get(key, 0.0) * scale
                eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

        eff_nbrs, eff_deg = build_neighbors(eff_weights)

        alpha_start = 2.0
        alpha_end = 1.0
        ls_rounds = 6 if rsdiwr_iter == 0 else 4

        cur_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights, alpha_exp=alpha_start, max_rounds=ls_rounds)

        static_cost = compute_cost(cur_m, static_weight)
        if static_cost < best_cost:
            best_cost = static_cost
            best_m, best_rm = list(cur_m), list(cur_rm)

        # Reset bandit
        for k in range(N_MODES):
            mode_successes[k] = 0
            mode_attempts[k] = 0

        T = max(cur_cost_nl * 0.05, 1.0)
        T_init = T
        alpha_sa = 0.94
        reheat_interval = max(n_ils // 4, 8)
        sub_qap_interval = 50

        for ils_iter in range(n_ils):
            if _time() - t_start > time_budget:
                break

            progress = ils_iter / max(n_ils - 1, 1)
            alpha_exp = alpha_start + (alpha_end - alpha_start) * progress

            saved_m = list(cur_m)
            saved_rm = list(cur_rm)
            saved_cost = cur_cost_nl

            mode = select_mode()
            perturbation_modes[mode](cur_m, cur_rm, nbrs=eff_nbrs, weights=eff_weights, alpha_exp=alpha_exp)

            new_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights, alpha_exp=alpha_exp, max_rounds=3)

            improvement = saved_cost - new_cost_nl
            success = improvement > 1e-12

            if improvement > 0:
                cur_cost_nl = new_cost_nl
                sc = compute_cost(cur_m, static_weight)
                if sc < best_cost:
                    best_cost = sc
                    best_m, best_rm = list(cur_m), list(cur_rm)
            elif random.random() < math.exp(min(0, improvement / max(T, 1e-10))):
                cur_cost_nl = new_cost_nl
            else:
                cur_m[:] = saved_m
                cur_rm[:] = saved_rm
                cur_cost_nl = saved_cost

            mode_attempts[mode] += 1
            if success:
                mode_successes[mode] += 1

            T *= alpha_sa
            if (ils_iter + 1) % reheat_interval == 0:
                T = max(T, T_init * 0.35)

            # --- Sub-QAP exact rotation on RSDIWR-identified hotspots ---
            if ils_iter > 0 and ils_iter % sub_qap_interval == 0 and swap_involvement and len(interacting_logical) >= 4:
                K_sub = min(8, len(interacting_logical))
                if K_sub >= 3:
                    # Top-K logical qubits by cumulative swap involvement
                    sorted_hotspot = sorted(interacting_logical, key=lambda lq: -swap_involvement.get(lq, 0.0))
                    hotspot_lqs = sorted_hotspot[:K_sub]

                    result = solve_sub_qap_exact(hotspot_lqs, cur_m, cur_rm, eff_weights)
                    if result is not None:
                        new_m, new_rm = result
                        new_sc = compute_cost(new_m, static_weight)
                        if new_sc < compute_cost(cur_m, static_weight):
                            cur_m[:] = new_m
                            cur_rm[:] = new_rm
                            cur_cost_nl = compute_cost_nonlinear(cur_m, eff_weights, alpha_exp)
                            if new_sc < best_cost:
                                best_cost = new_sc
                                best_m, best_rm = list(cur_m), list(cur_rm)

        # Routing simulation for next RSDIWR round
        if rsdiwr_iter < n_rsdiwr - 1:
            sim_depth = 12 + rsdiwr_iter * 6
            swap_involvement = simulate_routing(best_m, best_rm, max_layers=sim_depth)

        cur_m = list(best_m)
        cur_rm = list(best_rm)
        cur_cost_nl = compute_cost_nonlinear(cur_m, eff_weights, alpha_end)

    # ---------------------------------------------------------------
    # Step 14: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)