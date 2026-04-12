def init_mapping(self):
    import math
    import random
    from collections import defaultdict, deque
    from time import time as _time

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    dist = self.distance_matrix

    # ---------------------------------------------------------------
    # Step 1: Build DAG and topological layers
    # ---------------------------------------------------------------
    all_gates = sorted(self.access.keys())
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]

    last_gate_on_qubit = {}
    successors_dag = defaultdict(set)
    predecessors_dag = defaultdict(set)
    for g in all_gates:
        for q in self.access[g]:
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                successors_dag[pred].add(g)
                predecessors_dag[g].add(pred)
            last_gate_on_qubit[q] = g

    in_degree = {g: len(predecessors_dag[g]) for g in all_gates}
    gate_layer = {g: 0 for g in all_gates}
    temp_in = dict(in_degree)
    queue = deque(g for g in all_gates if in_degree[g] == 0)
    topo_order = []
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in successors_dag[g]:
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)

    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors_dag[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # 2q DAG for routing simulation
    gates_2q = {}
    dag2q_succ = defaultdict(set)
    dag2q_pred = defaultdict(set)
    last_2q_on_qubit = {}
    logical_qubits_set = set()

    for gate in all_gates:
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            gates_2q[gate] = (q1, q2)
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            for q in [q1, q2]:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != gate:
                        dag2q_succ[prev].add(gate)
                        dag2q_pred[gate].add(prev)
                last_2q_on_qubit[q] = gate
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)

    if not gates_2q:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Topological order for 2q DAG
    topo_2q = []
    in_deg_2q = {g: len(dag2q_pred[g]) for g in gates_2q}
    q2 = deque(sorted(g for g in gates_2q if in_deg_2q[g] == 0))
    while q2:
        g = q2.popleft()
        topo_2q.append(g)
        for s in dag2q_succ[g]:
            in_deg_2q[s] -= 1
            if in_deg_2q[s] == 0:
                q2.append(s)

    dep_count = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count[g] = max(dep_count[g], dep_count[s] + 1)

    # ---------------------------------------------------------------
    # Step 2: Interaction weights — critical-path weighting (from P1)
    # ---------------------------------------------------------------
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)

    static_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        cp = critical_path[g] + 1
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
        for pq2 in self.backend.get(pq, []):
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

    def compute_cost_nonlinear(m, weights, alpha_exp):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = dist[m[q1]][m[q2]]
                cost += w * (d ** alpha_exp)
        return cost

    def compute_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * dist[m[q1]][m[q2]]
        return cost

    def delta_swap_cost_nonlinear(m, rm, pq_a, pq_b, nbrs, alpha_exp):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        delta = 0.0
        affected = set()
        if lq_a in nbrs:
            affected.update(nbrs[lq_a].keys())
        if lq_b in nbrs:
            affected.update(nbrs[lq_b].keys())
        for q in affected:
            if q == lq_a or q == lq_b:
                continue
            pq_q = m[q]
            w_a = nbrs.get(lq_a, {}).get(q, 0.0)
            if w_a > 0:
                old_d = dist[pq_a][pq_q] ** alpha_exp
                new_d = dist[pq_b][pq_q] ** alpha_exp
                delta += w_a * (new_d - old_d)
            w_b = nbrs.get(lq_b, {}).get(q, 0.0)
            if w_b > 0:
                old_d = dist[pq_b][pq_q] ** alpha_exp
                new_d = dist[pq_a][pq_q] ** alpha_exp
                delta += w_b * (new_d - old_d)
        return delta

    def do_swap(m, rm, pq_a, pq_b):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        m[lq_a], m[lq_b] = pq_b, pq_a
        rm[pq_a], rm[pq_b] = lq_b, lq_a

    # ---------------------------------------------------------------
    # Step 5: Conflict-Graph-Guided MIS Seed (from P2, enhanced
    #   with P1's adjacency bonus scaling formula)
    # ---------------------------------------------------------------
    static_nbrs, static_deg = build_neighbors(static_weight)
    max_iw = max(static_weight.values()) if static_weight else 1.0

    def conflict_graph_seed(start_pq):
        """
        Build conflict graph via Jaccard similarity of top-3 interaction
        partners. Find greedy MIS. Place MIS qubits first (non-competing),
        then remaining. Uses P1's scaled adjacency bonus (0.90 - 0.10*iw/max_iw).
        """
        if len(interacting_logical) < 2:
            return None, None

        # Top-3 most-interacting partners per qubit
        top_partners = {}
        for lq in interacting_logical:
            nbr_list = sorted(static_nbrs.get(lq, {}).items(),
                              key=lambda x: x[1], reverse=True)
            top_partners[lq] = set(p for p, _ in nbr_list[:3])

        # Build conflict graph: edge if Jaccard(top-3) > 0.5
        conflict_adj = defaultdict(set)
        il = interacting_logical
        for i in range(len(il)):
            for j in range(i + 1, len(il)):
                l1, l2 = il[i], il[j]
                s1 = top_partners.get(l1, set())
                s2 = top_partners.get(l2, set())
                if not s1 and not s2:
                    continue
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                if union > 0 and intersection / union > 0.5:
                    conflict_adj[l1].add(l2)
                    conflict_adj[l2].add(l1)

        # Greedy MIS in descending interaction-degree order
        sorted_by_deg = sorted(interacting_logical,
                               key=lambda q: logical_degree[q], reverse=True)
        mis = set()
        excluded = set()
        for lq in sorted_by_deg:
            if lq not in excluded:
                mis.add(lq)
                for neighbor in conflict_adj.get(lq, set()):
                    excluded.add(neighbor)

        non_mis = [lq for lq in sorted_by_deg if lq not in mis]
        mis_ordered = sorted(mis, key=lambda q: logical_degree[q], reverse=True)

        m = [-1] * num_q
        rm = [-1] * num_q
        used_phys = set()

        # Place MIS qubits first — they don't compete for same region
        if mis_ordered:
            first_lq = mis_ordered[0]
            m[first_lq] = start_pq
            rm[start_pq] = first_lq
            used_phys.add(start_pq)

            for lq in mis_ordered[1:]:
                nbrs_placed = {}
                for plq in mis:
                    if m[plq] >= 0 and plq in static_nbrs.get(lq, {}):
                        nbrs_placed[plq] = static_nbrs[lq][plq]

                if nbrs_placed:
                    best_pq = None
                    best_score = float('inf')
                    for pq in physical_qubits:
                        if pq in used_phys:
                            continue
                        score = 0.0
                        for plq, iw in nbrs_placed.items():
                            d = dist[pq][m[plq]]
                            cost = iw * d
                            # P1's scaled adjacency bonus
                            if m[plq] in hw_adj[pq]:
                                cost *= 0.90 - 0.10 * (iw / max_iw)
                            score += cost
                        if score < best_score:
                            best_score = score
                            best_pq = pq
                else:
                    best_pq = None
                    best_score = float('inf')
                    for pq in physical_qubits:
                        if pq not in used_phys:
                            if phys_centrality[pq] < best_score:
                                best_score = phys_centrality[pq]
                                best_pq = pq

                if best_pq is not None:
                    m[lq] = best_pq
                    rm[best_pq] = lq
                    used_phys.add(best_pq)

        # Place non-MIS qubits using P1's degree-rank tie-breaking
        placed_set = set(lq for lq in interacting_logical if m[lq] >= 0)

        for lq in non_mis:
            nbrs_placed = {}
            for plq in placed_set:
                w = static_nbrs.get(lq, {}).get(plq, 0.0)
                if w > 0:
                    nbrs_placed[plq] = w

            if nbrs_placed:
                near_ties = []
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        d = dist[pq][m[plq]]
                        cost = iw * d
                        if m[plq] in hw_adj[pq]:
                            cost *= 0.90 - 0.10 * (iw / max_iw)
                        score += cost
                    near_ties.append((score, pq))
                    if score < best_score:
                        best_score = score

                if near_ties and best_score > 0:
                    threshold = best_score * 1.05
                    candidates_list = [(s, pq) for s, pq in near_ties if s <= threshold]
                    if len(candidates_list) > 1 and lq in logical_degree_rank:
                        lq_rn = logical_degree_rank[lq] / max_logical_rank
                        best_pq = min(candidates_list, key=lambda x: (
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rn), x[0]))[1]
                    else:
                        best_pq = min(candidates_list, key=lambda x: x[0])[1]
                elif near_ties:
                    best_pq = min(near_ties, key=lambda x: x[0])[1]
                else:
                    best_pq = None
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_phys:
                        if phys_centrality[pq] < best_score:
                            best_score = phys_centrality[pq]
                            best_pq = pq

            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                used_phys.add(best_pq)
                placed_set.add(lq)

        # Place remaining non-interacting logical qubits
        remaining_lqs = [lq for lq in logical_qubits if m[lq] == -1]
        for lq in remaining_lqs:
            best_pq = None
            best_score = float('inf')
            for pq in physical_qubits:
                if pq not in used_phys:
                    if phys_centrality[pq] < best_score:
                        best_score = phys_centrality[pq]
                        best_pq = pq
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                used_phys.add(best_pq)

        return m, rm

    # ---------------------------------------------------------------
    # Step 6: Spectral ordering seed (for diversity)
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
                new_v = [0.0] * n
                for i in range(n):
                    for j in range(n):
                        new_v[i] += mat[i][j] * v[j]
                norm = math.sqrt(sum(x * x for x in new_v)) or 1e-12
                v = [x / norm for x in new_v]
            return v

        max_diag = max(L[i][i] for i in range(n)) + 1.0
        shifted = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                shifted[i][j] = -L[i][j]
            shifted[i][i] += max_diag

        v1 = power_iter(shifted, 150)

        dot_v1 = sum(x * x for x in v1) or 1e-12
        lam1_shifted = sum(v1[i] * sum(shifted[i][j] * v1[j] for j in range(n)) for i in range(n)) / dot_v1
        shifted2 = [row[:] for row in shifted]
        for i in range(n):
            for j in range(n):
                shifted2[i][j] -= lam1_shifted * v1[i] * v1[j] / dot_v1

        fiedler = power_iter(shifted2, 200)

        fiedler_order = sorted(range(n), key=lambda i: fiedler[i])
        sorted_logical = [interacting_logical[i] for i in fiedler_order]

        phys_sorted = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
        start_pq = phys_sorted[0]
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
        n_phys = len(bfs_order)
        offset = max(0, (n_phys - n_interact) // 2)
        for i, lq in enumerate(sorted_logical):
            pq = bfs_order[min(offset + i, n_phys - 1)]
            if rm[pq] != -1:
                best_pq = None
                best_d = float('inf')
                for cpq in physical_qubits:
                    if rm[cpq] == -1:
                        d = dist[pq][cpq]
                        if d < best_d:
                            best_d = d
                            best_pq = cpq
                pq = best_pq
            m[lq] = pq
            rm[pq] = lq

        return m, rm

    # ---------------------------------------------------------------
    # Step 7: Greedy placement (P1's adjacency bonus + degree-rank
    #         tie-breaking)
    # ---------------------------------------------------------------
    sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)
    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    def run_greedy_placement(start_lq, start_pq, nbrs, deg):
        used_phys = {start_pq}
        m = [-1] * num_q
        rm = [-1] * num_q
        m[start_lq] = start_pq
        rm[start_pq] = start_lq
        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            best_lq, best_w = None, -1.0
            for lq in remaining:
                w = sum(nbrs.get(lq, {}).get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            nbrs_placed = {plq: nbrs.get(best_lq, {}).get(plq, 0.0)
                           for plq in placed if plq in nbrs.get(best_lq, {})}

            if nbrs_placed:
                near_ties = []
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        d = dist[pq][m[plq]]
                        cost = iw * d
                        if m[plq] in hw_adj[pq]:
                            cost *= 0.90 - 0.10 * (iw / max_iw)
                        score += cost
                    near_ties.append((score, pq))
                    if score < best_score:
                        best_score = score

                if near_ties and best_score > 0:
                    threshold = best_score * 1.05
                    candidates_list = [(s, pq) for s, pq in near_ties if s <= threshold]
                    if len(candidates_list) > 1 and best_lq in logical_degree_rank:
                        lq_rn = logical_degree_rank[best_lq] / max_logical_rank
                        best_pq = min(candidates_list, key=lambda x: (
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rn), x[0]))[1]
                    else:
                        best_pq = min(candidates_list, key=lambda x: x[0])[1]
                elif near_ties:
                    best_pq = min(near_ties, key=lambda x: x[0])[1]
                else:
                    best_pq = None
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_phys:
                        score = phys_centrality[pq]
                        if score < best_score:
                            best_score = score
                            best_pq = pq

            m[best_lq] = best_pq
            rm[best_pq] = best_lq
            used_phys.add(best_pq)
            placed.add(best_lq)
            remaining.discard(best_lq)
        return m, rm

    def fill_unmapped(m, rm):
        unmapped = [q for q in range(num_q) if m[q] == -1]
        free = [pq for pq in range(num_q) if rm[pq] == -1]
        for lq, pq in zip(unmapped, free):
            m[lq] = pq
            rm[pq] = lq

    # ---------------------------------------------------------------
    # Step 8: Local search with nonlinear distance penalty
    # ---------------------------------------------------------------
    def local_search(m, rm, nbrs, weights, alpha_exp, max_rounds=5):
        if len(interacting_logical) <= 1:
            return compute_cost_nonlinear(m, weights, alpha_exp)
        for _ in range(max_rounds):
            improved = False
            best_d, best_pair = 0.0, None
            active_phys = [m[q] for q in interacting_logical]
            for pq1 in active_phys:
                for pq2 in hw_adj.get(pq1, set()):
                    d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                    if d < best_d:
                        best_d = d
                        best_pair = (pq1, pq2)
            n_random = min(150, len(interacting_logical) * 3)
            for _ in range(n_random):
                i, j = random.sample(range(len(interacting_logical)), 2)
                pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                if d < best_d:
                    best_d = d
                    best_pair = (pq1, pq2)
            if best_pair and best_d < -1e-12:
                do_swap(m, rm, *best_pair)
                improved = True
            if not improved:
                for i in range(len(interacting_logical)):
                    for j in range(i + 1, len(interacting_logical)):
                        pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                        d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                        if d < best_d:
                            best_d = d
                            best_pair = (pq1, pq2)
                if best_pair and best_d < -1e-12:
                    do_swap(m, rm, *best_pair)
                else:
                    break
        return compute_cost_nonlinear(m, weights, alpha_exp)

    # ---------------------------------------------------------------
    # Step 9: Routing simulation for RSDIWR feedback
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=20):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)

        if not gates_2q:
            return swap_counts

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
                for nb in self.backend.get(pq, []):
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
                    best_score = score
                    best_swap = (s1, s2)

            if best_swap is None:
                break

            s1, s2 = best_swap
            l1, l2 = sim_rm[s1], sim_rm[s2]
            sim_m[l1], sim_m[l2] = s2, s1
            sim_rm[s1], sim_rm[s2] = l2, l1

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0

        return swap_counts

    # ---------------------------------------------------------------
    # Step 10: Five perturbation modes with bandit selection
    # ---------------------------------------------------------------
    def perturb_random(m, rm, **kw):
        if len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    def perturb_segment_shuffle(m, rm, **kw):
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

    def perturb_worst_pair(m, rm, weights=None, alpha_exp=1.5, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        pair_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = dist[m[q1]][m[q2]]
                c = w * (d ** alpha_exp)
                pair_costs.append((c, q1, q2))
        if not pair_costs:
            return
        pair_costs.sort(reverse=True)
        top_n = min(3, len(pair_costs))
        _, tq1, tq2 = pair_costs[random.randint(0, top_n - 1)]
        pq1, pq2 = m[tq1], m[tq2]
        adj_of_pq2 = list(hw_adj.get(pq2, set()))
        if adj_of_pq2:
            target = random.choice(adj_of_pq2)
            if target != pq1:
                do_swap(m, rm, pq1, target)
            else:
                do_swap(m, rm, pq1, pq2)
        else:
            do_swap(m, rm, pq1, pq2)

    def perturb_edge_targeted(m, rm, weights=None, alpha_exp=1.5, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        edge_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c = w * (dist[m[q1]][m[q2]] ** alpha_exp)
                edge_costs.append((c, q1, q2))
        if not edge_costs:
            return
        edge_costs.sort(reverse=True)
        top_n = min(5, len(edge_costs))
        _, eq1, eq2 = edge_costs[random.randint(0, top_n - 1)]
        pq1, pq2 = m[eq1], m[eq2]
        neighbors_of_pq2 = list(hw_adj.get(pq2, set()))
        if neighbors_of_pq2:
            target = random.choice(neighbors_of_pq2)
            if target != pq1:
                do_swap(m, rm, pq1, target)
            else:
                do_swap(m, rm, pq1, pq2)
        else:
            do_swap(m, rm, pq1, pq2)

    def perturb_lns(m, rm, nbrs=None, alpha_exp=1.5, **kw):
        if nbrs is None or len(interacting_logical) < 3:
            perturb_random(m, rm)
            return
        k = min(random.randint(3, 8), len(interacting_logical))
        qcost = {}
        for lq in interacting_logical:
            c = 0.0
            for partner, w in nbrs.get(lq, {}).items():
                if m[partner] >= 0 and m[lq] >= 0:
                    c += w * (dist[m[lq]][m[partner]] ** alpha_exp)
            qcost[lq] = c
        sorted_qs = sorted(qcost, key=lambda q: qcost[q], reverse=True)
        top_half = sorted_qs[:max(k, len(sorted_qs) // 2)]
        subset = random.sample(top_half, min(k, len(top_half)))

        freed_phys = []
        for lq in subset:
            freed_phys.append(m[lq])
            rm[m[lq]] = -1
            m[lq] = -1

        placed_set = set(lq for lq in interacting_logical if m[lq] >= 0)
        for lq in sorted(subset, key=lambda q: qcost[q], reverse=True):
            best_pq, best_sc = None, float('inf')
            for pq in freed_phys:
                if rm[pq] != -1:
                    continue
                sc = 0.0
                for partner, w in nbrs.get(lq, {}).items():
                    if partner in placed_set and m[partner] >= 0:
                        sc += w * (dist[pq][m[partner]] ** alpha_exp)
                if sc < best_sc:
                    best_sc = sc
                    best_pq = pq
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                placed_set.add(lq)
            else:
                for pq in freed_phys:
                    if rm[pq] == -1:
                        m[lq] = pq
                        rm[pq] = lq
                        placed_set.add(lq)
                        break

    perturbation_modes = [perturb_random, perturb_segment_shuffle,
                          perturb_worst_pair, perturb_edge_targeted, perturb_lns]
    K = len(perturbation_modes)

    WINDOW_SIZE = 100
    EPSILON = 0.1
    window = []
    mode_successes = [0] * K
    mode_attempts = [0] * K

    def select_perturbation_mode():
        if random.random() < EPSILON or sum(mode_attempts) < K * 2:
            return random.randint(0, K - 1)
        rates = []
        for k in range(K):
            if mode_attempts[k] > 0:
                rates.append(mode_successes[k] / mode_attempts[k])
            else:
                rates.append(1.0)
        total = sum(rates)
        if total < 1e-12:
            return random.randint(0, K - 1)
        r = random.random() * total
        cumul = 0.0
        for k in range(K):
            cumul += rates[k]
            if r <= cumul:
                return k
        return K - 1

    def update_window(mode_idx, success):
        window.append((mode_idx, success))
        mode_attempts[mode_idx] += 1
        if success:
            mode_successes[mode_idx] += 1
        if len(window) > WINDOW_SIZE:
            old_mode, old_success = window.pop(0)
            mode_attempts[old_mode] -= 1
            if old_success:
                mode_successes[old_mode] -= 1

    # ---------------------------------------------------------------
    # Step 11: Build initial mapping candidates
    #   Greedy seeds (P1) + Conflict-graph MIS seeds (P2) +
    #   Spectral seed + Random seed
    # ---------------------------------------------------------------
    candidates = []

    # Greedy seeds (from P1 — primary high-quality seeds)
    for s_lq in seed_lqs:
        for s_pq in seed_pqs:
            m, rm = run_greedy_placement(s_lq, s_pq, static_nbrs, static_deg)
            fill_unmapped(m, rm)
            cost = compute_cost(m, static_weight)
            candidates.append((cost, m, rm))

    # Conflict-graph MIS seeds (from P2 — structural diversity)
    for s_pq in seed_pqs:
        cg_m, cg_rm = conflict_graph_seed(s_pq)
        if cg_m is not None:
            fill_unmapped(cg_m, cg_rm)
            cost = compute_cost(cg_m, static_weight)
            candidates.append((cost, cg_m, cg_rm))

    # Spectral seed (for additional diversity)
    spec_m, spec_rm = spectral_seed()
    if spec_m is not None:
        fill_unmapped(spec_m, spec_rm)
        spec_cost = compute_cost(spec_m, static_weight)
        candidates.append((spec_cost, spec_m, spec_rm))

    # One random seed
    m_rand = [-1] * num_q
    rm_rand = [-1] * num_q
    shuffled_phys = list(physical_qubits)
    random.shuffle(shuffled_phys)
    for i, lq in enumerate(logical_qubits):
        m_rand[lq] = shuffled_phys[i]
        rm_rand[shuffled_phys[i]] = lq
    fill_unmapped(m_rand, rm_rand)
    c = compute_cost(m_rand, static_weight)
    candidates.append((c, m_rand, rm_rand))

    if not candidates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    candidates.sort(key=lambda x: x[0])

    # Refine top 5 candidates with nonlinear local search
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
            best_m = list(m_c)
            best_rm = list(rm_c)

    # ---------------------------------------------------------------
    # Step 12: Dual-Phase ILS with RSDIWR
    # ---------------------------------------------------------------
    time_budget = 25.0
    t_start = _time()
    n_rsdiwr = 4

    cur_m = list(best_m)
    cur_rm = list(best_rm)
    swap_counts = defaultdict(float)

    for rsdiwr_iter in range(n_rsdiwr):
        elapsed = _time() - t_start
        if elapsed > time_budget:
            break

        # Time-adaptive ILS iterations
        remaining_time = time_budget - elapsed
        remaining_iters = n_rsdiwr - rsdiwr_iter
        time_for_this_iter = remaining_time / max(remaining_iters, 1)
        n_ils = max(55, int(num_q * time_for_this_iter / 0.5)) if time_for_this_iter > 1.0 else max(30, num_q // 3)

        # Blend static weights with routing feedback
        if rsdiwr_iter == 0 or not swap_counts:
            eff_weights = dict(static_weight)
        else:
            max_sw = max(swap_counts.values()) if swap_counts else 1.0
            scale = max(static_weight.values()) / max(max_sw, 1e-10)
            alpha_blend = max(0.3, 1.0 - 0.3 * rsdiwr_iter)
            eff_weights = defaultdict(float)
            all_keys = set(static_weight.keys()) | set(swap_counts.keys())
            for key in all_keys:
                w_s = static_weight.get(key, 0.0)
                w_r = swap_counts.get(key, 0.0) * scale
                eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

        eff_nbrs, eff_deg = build_neighbors(eff_weights)

        alpha_start = 2.0
        alpha_end = 1.0

        ls_rounds = 6 if rsdiwr_iter == 0 else 4
        cur_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights,
                                   alpha_exp=alpha_start, max_rounds=ls_rounds)

        static_cost = compute_cost(cur_m, static_weight)
        if static_cost < best_cost:
            best_cost = static_cost
            best_m = list(cur_m)
            best_rm = list(cur_rm)

        # Reset bandit window for new weight landscape
        window.clear()
        for k in range(K):
            mode_successes[k] = 0
            mode_attempts[k] = 0

        T = max(cur_cost_nl * 0.05, 1.0)
        T_init = T
        alpha_sa = 0.94
        reheat_interval = max(n_ils // 4, 8)

        for ils_iter in range(n_ils):
            if _time() - t_start > time_budget:
                break

            progress = ils_iter / max(n_ils - 1, 1)
            alpha_exp = alpha_start + (alpha_end - alpha_start) * progress

            saved_m = list(cur_m)
            saved_rm = list(cur_rm)
            saved_cost = cur_cost_nl

            mode = select_perturbation_mode()
            perturbation_modes[mode](cur_m, cur_rm,
                                     nbrs=eff_nbrs, weights=eff_weights,
                                     alpha_exp=alpha_exp)

            new_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights,
                                       alpha_exp=alpha_exp, max_rounds=3)

            success = new_cost_nl < saved_cost - 1e-12

            improvement = saved_cost - new_cost_nl
            if improvement > 0:
                cur_cost_nl = new_cost_nl
                sc = compute_cost(cur_m, static_weight)
                if sc < best_cost:
                    best_cost = sc
                    best_m = list(cur_m)
                    best_rm = list(cur_rm)
            elif random.random() < math.exp(min(0, improvement / max(T, 1e-10))):
                cur_cost_nl = new_cost_nl
            else:
                cur_m[:] = saved_m
                cur_rm[:] = saved_rm
                cur_cost_nl = saved_cost

            update_window(mode, success)
            T *= alpha_sa
            if (ils_iter + 1) % reheat_interval == 0:
                T = max(T, T_init * 0.35)

        # Routing simulation with ramped depth
        if rsdiwr_iter < n_rsdiwr - 1:
            sim_depth = 12 + rsdiwr_iter * 6
            swap_counts = simulate_routing(best_m, best_rm, max_layers=sim_depth)

        cur_m = list(best_m)
        cur_rm = list(best_rm)
        cur_cost_nl = compute_cost_nonlinear(cur_m, eff_weights, alpha_end)

    # ---------------------------------------------------------------
    # Step 13: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)