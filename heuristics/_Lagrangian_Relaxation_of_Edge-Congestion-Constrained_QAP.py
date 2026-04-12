def init_mapping(self):
    import math
    import numpy as np
    from collections import defaultdict, deque
    from time import time as _time

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    dist = self.distance_matrix

    # ---------------------------------------------------------------
    # Step 1: Build DAG, topological sort, critical path, gate layers
    # ---------------------------------------------------------------
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in sorted(self.access.keys()):
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
    topo_order = []
    topo_rank = {}
    gate_layer = {g: 0 for g in all_gates}
    rank = 0
    temp_in = dict(in_degree)

    while queue:
        g = queue.popleft()
        topo_order.append(g)
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # ---------------------------------------------------------------
    # Step 2: Interaction weights (critical-path based)
    # ---------------------------------------------------------------
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)

    logical_qubits_set = set()
    static_weight = defaultdict(float)
    logical_degree = defaultdict(float)
    gate_pair_map = defaultdict(list)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        logical_qubits_set.add(q1)
        logical_qubits_set.add(q2)
        key = (min(q1, q2), max(q1, q2))
        cp = critical_path[g] + 1
        layer = gate_layer[g]
        w = cp * (max_layer - layer + 1)
        static_weight[key] += w
        logical_degree[q1] += w
        logical_degree[q2] += w
        gate_pair_map[key].append(g)

    for g in all_gates:
        if len(self.access[g]) == 1:
            logical_qubits_set.add(self.access[g][0])

    logical_qubits = sorted(logical_qubits_set)
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    # ---------------------------------------------------------------
    # Step 3: Precompute physical graph properties + shortest paths
    # ---------------------------------------------------------------
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(dist[pq][pq2] for pq2 in physical_qubits)

    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            hw_adj[pq].add(pq2)

    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}

    # Precompute shortest path edges for each pair of physical qubits
    def get_shortest_path_edges(p1, p2):
        if p1 == p2:
            return []
        parent = {p1: None}
        bfs_q = deque([p1])
        while bfs_q:
            cur = bfs_q.popleft()
            if cur == p2:
                break
            for nb in self.backend.get(cur, []):
                if nb not in parent:
                    parent[nb] = cur
                    bfs_q.append(nb)
        edges = []
        cur = p2
        while parent.get(cur) is not None:
            prev = parent[cur]
            edges.append((min(prev, cur), max(prev, cur)))
            cur = prev
        return edges

    path_edges_cache = {}
    for p1 in physical_qubits:
        for p2 in physical_qubits:
            if p1 < p2:
                path_edges_cache[(p1, p2)] = get_shortest_path_edges(p1, p2)

    def get_path_edges(p1, p2):
        if p1 == p2:
            return []
        key = (min(p1, p2), max(p1, p2))
        return path_edges_cache.get(key, [])

    hw_edges = set()
    for pq in physical_qubits:
        for nb in self.backend.get(pq, []):
            hw_edges.add((min(pq, nb), max(pq, nb)))

    # ---------------------------------------------------------------
    # Step 4: Seed generation
    # ---------------------------------------------------------------
    if interacting_logical:
        sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
        seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    else:
        seed_lqs = logical_qubits[:1] if logical_qubits else []

    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)

    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    # MIS seed
    def build_mis_seed():
        if len(interacting_logical) < 3:
            return None
        top_partners = defaultdict(list)
        for (q1, q2), w in static_weight.items():
            top_partners[q1].append((w, q2))
            top_partners[q2].append((w, q1))
        for q in top_partners:
            top_partners[q].sort(reverse=True)
            top_partners[q] = top_partners[q][:3]
        partner_sets = {}
        for q in interacting_logical:
            partner_sets[q] = set(p for _, p in top_partners.get(q, []))

        conflict = defaultdict(set)
        for i, q1 in enumerate(interacting_logical):
            for q2 in interacting_logical[i + 1:]:
                s1 = partner_sets.get(q1, set())
                s2 = partner_sets.get(q2, set())
                if s1 and s2:
                    jaccard = len(s1 & s2) / max(len(s1 | s2), 1)
                    if jaccard > 0.5:
                        conflict[q1].add(q2)
                        conflict[q2].add(q1)

        order = sorted(interacting_logical, key=lambda q: (len(conflict.get(q, set())), -logical_degree.get(q, 0)))
        mis = []
        excluded = set()
        for q in order:
            if q not in excluded:
                mis.append(q)
                excluded.update(conflict.get(q, set()))
        return mis

    # Spectral seed
    def build_spectral_order():
        if len(interacting_logical) < 3:
            return None
        idx_map = {q: i for i, q in enumerate(interacting_logical)}
        n = len(interacting_logical)
        laplacian = np.zeros((n, n))
        for (q1, q2), w in static_weight.items():
            if q1 in idx_map and q2 in idx_map:
                i, j = idx_map[q1], idx_map[q2]
                laplacian[i][i] += w
                laplacian[j][j] += w
                laplacian[i][j] -= w
                laplacian[j][i] -= w

        rng = np.random.RandomState(42)
        v = rng.randn(n)
        v /= np.linalg.norm(v)

        for _ in range(100):
            v_new = laplacian @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-15:
                break
            v = v_new / norm

        lam1 = v @ (laplacian @ v)
        fiedler = rng.randn(n)
        fiedler -= fiedler.mean()
        fiedler /= max(np.linalg.norm(fiedler), 1e-15)

        for _ in range(200):
            f_new = laplacian @ fiedler
            f_new -= (f_new @ v) * v
            f_new -= f_new.mean()
            norm = np.linalg.norm(f_new)
            if norm < 1e-15:
                break
            fiedler = f_new / norm

        order = sorted(range(n), key=lambda i: fiedler[i])
        return [interacting_logical[i] for i in order]

    # ---------------------------------------------------------------
    # Step 5: Build 2q DAG for routing simulation
    # ---------------------------------------------------------------
    last_2q_on_qubit = {}
    gates_2q = {}
    dag2q_succ = defaultdict(set)
    dag2q_pred = defaultdict(set)

    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            gates_2q[gate] = (q1, q2)
            for q in [q1, q2]:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != gate:
                        dag2q_succ[prev].add(gate)
                        dag2q_pred[gate].add(prev)
                last_2q_on_qubit[q] = gate

    topo_2q = []
    in_deg_2q = {g: len(dag2q_pred[g]) for g in gates_2q}
    q2_queue = deque(sorted(g for g in gates_2q if in_deg_2q[g] == 0))
    while q2_queue:
        g = q2_queue.popleft()
        topo_2q.append(g)
        for s in dag2q_succ[g]:
            in_deg_2q[s] -= 1
            if in_deg_2q[s] == 0:
                q2_queue.append(s)

    dep_count_2q = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count_2q[g] += dep_count_2q[s] + 1

    # ---------------------------------------------------------------
    # Step 6: Helper functions
    # ---------------------------------------------------------------
    def build_neighbors_from_weights(weights):
        neighbors = defaultdict(dict)
        degree = defaultdict(float)
        for (q1, q2), w in weights.items():
            neighbors[q1][q2] = w
            neighbors[q2][q1] = w
            degree[q1] += w
            degree[q2] += w
        return neighbors, degree

    def run_greedy_placement(start_lq, start_pq, logical_nbrs, eff_degree, eff_dist=None):
        d = eff_dist if eff_dist is not None else dist
        max_iw = max((w for nbrs in logical_nbrs.values() for w in nbrs.values()), default=1.0)

        def dyn_bonus(iw):
            return 0.90 - 0.10 * (iw / max_iw)

        used_phys = {start_pq}
        m = [-1] * num_q
        rm = [-1] * num_q
        m[start_lq] = start_pq
        rm[start_pq] = start_lq
        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            best_lq = None
            best_w = -1.0
            for lq in remaining:
                w = sum(logical_nbrs[lq].get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            nbrs_placed = {plq: logical_nbrs[best_lq].get(plq, 0.0)
                           for plq in placed if plq in logical_nbrs[best_lq]}

            if nbrs_placed:
                near_ties = []
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        dd = d[pq][m[plq]]
                        cost = iw * dd
                        if m[plq] in hw_adj[pq]:
                            cost *= dyn_bonus(iw)
                        score += cost
                    near_ties.append((score, pq))
                    if score < best_score:
                        best_score = score

                if near_ties and best_score > 0:
                    threshold = best_score * 1.05
                    candidates = [(s, pq) for s, pq in near_ties if s <= threshold]
                    if len(candidates) > 1 and best_lq in logical_degree_rank:
                        lq_rank_norm = logical_degree_rank[best_lq] / max_logical_rank
                        best_pq = min(candidates, key=lambda x: (
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rank_norm),
                            x[0]
                        ))[1]
                    else:
                        best_pq = min(candidates, key=lambda x: x[0])[1]
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

    def compute_total_cost(m, weights, eff_dist=None):
        d = eff_dist if eff_dist is not None else dist
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * d[m[q1]][m[q2]]
        return cost

    def run_swap_refinement(m, rm, logical_nbrs, max_rounds, eff_dist=None):
        d = eff_dist if eff_dist is not None else dist
        if len(interacting_logical) <= 1:
            return
        improved = True
        round_count = 0
        while improved and round_count < max_rounds:
            improved = False
            round_count += 1
            for i in range(len(interacting_logical)):
                for j in range(i + 1, len(interacting_logical)):
                    lq_a = interacting_logical[i]
                    lq_b = interacting_logical[j]
                    pq_a = m[lq_a]
                    pq_b = m[lq_b]

                    delta = 0.0
                    affected = set()
                    affected.update(logical_nbrs[lq_a].keys())
                    affected.update(logical_nbrs[lq_b].keys())

                    for q in affected:
                        if q == lq_a or q == lq_b:
                            continue
                        pq_q = m[q]
                        w_a = logical_nbrs[lq_a].get(q, 0.0)
                        if w_a > 0:
                            delta += w_a * (d[pq_b][pq_q] - d[pq_a][pq_q])
                        w_b = logical_nbrs[lq_b].get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (d[pq_a][pq_q] - d[pq_b][pq_q])

                    if delta < -1e-12:
                        m[lq_a] = pq_b
                        m[lq_b] = pq_a
                        rm[pq_a] = lq_b
                        rm[pq_b] = lq_a
                        improved = True

    def perturb_and_refine_sa(m, rm, weights, logical_nbrs, num_perturbations=9, eff_dist=None):
        d = eff_dist if eff_dist is not None else dist
        if len(interacting_logical) < 4:
            return m, rm
        best_m = list(m)
        best_rm = list(rm)
        best_cost = compute_total_cost(m, weights, eff_dist)
        current_m = list(m)
        current_rm = list(rm)
        current_cost = best_cost

        rng_seed = int(best_cost * 1000) % (2**31)
        rng = np.random.RandomState(rng_seed)

        T_init = max(best_cost * 0.02, 1.0)
        T_min = T_init * 0.01

        for p_idx in range(num_perturbations):
            m_try = list(current_m)
            rm_try = list(current_rm)
            mode = p_idx % 4
            T = T_init * (T_min / T_init) ** (p_idx / max(num_perturbations - 1, 1))

            if mode == 0 and len(interacting_logical) >= 4:
                edge_costs = []
                for (eq1, eq2), w in weights.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * d[m_try[eq1]][m_try[eq2]]
                        edge_costs.append((c, eq1, eq2))
                edge_costs.sort(reverse=True)
                for _, eq1, eq2 in edge_costs[:min(3, len(edge_costs))]:
                    pq1 = m_try[eq1]
                    pq2 = m_try[eq2]
                    current_dd = d[pq1][pq2]
                    best_delta = 0.0
                    best_swap_pair = None
                    for adj_pq in hw_adj[pq1]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq2:
                            continue
                        new_dd = d[adj_pq][pq2]
                        dd = new_dd - current_dd
                        if dd < best_delta:
                            best_delta = dd
                            best_swap_pair = (eq1, occ, pq1, adj_pq)
                    for adj_pq in hw_adj[pq2]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq1:
                            continue
                        new_dd = d[pq1][adj_pq]
                        dd = new_dd - current_dd
                        if dd < best_delta:
                            best_delta = dd
                            best_swap_pair = (eq2, occ, pq2, adj_pq)
                    if best_swap_pair is not None:
                        lq_a, lq_b, pq_a, pq_b = best_swap_pair
                        m_try[lq_a] = pq_b
                        m_try[lq_b] = pq_a
                        rm_try[pq_a] = lq_b
                        rm_try[pq_b] = lq_a
                        break
                indices = rng.choice(len(interacting_logical), size=2, replace=False)
                lq_c = interacting_logical[indices[0]]
                lq_d = interacting_logical[indices[1]]
                pq_c, pq_d = m_try[lq_c], m_try[lq_d]
                m_try[lq_c] = pq_d
                m_try[lq_d] = pq_c
                rm_try[pq_c] = lq_d
                rm_try[pq_d] = lq_c

            elif mode == 1 and len(interacting_logical) >= 4:
                qcost = defaultdict(float)
                for (eq1, eq2), w in weights.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * d[m_try[eq1]][m_try[eq2]]
                        qcost[eq1] += c
                        qcost[eq2] += c
                cost_pairs = [(qcost.get(q, 0.0), q) for q in interacting_logical]
                cost_pairs.sort(reverse=True)
                swapped = False
                for _, lq_a in cost_pairs[:min(3, len(cost_pairs))]:
                    pq_a = m_try[lq_a]
                    adj_candidates = []
                    for adj_pq in hw_adj[pq_a]:
                        occ = rm_try[adj_pq]
                        if occ >= 0:
                            adj_candidates.append((qcost.get(occ, 0.0), occ, adj_pq))
                    if adj_candidates:
                        adj_candidates.sort()
                        lq_b = adj_candidates[0][1]
                        pq_b = adj_candidates[0][2]
                        m_try[lq_a] = pq_b
                        m_try[lq_b] = pq_a
                        rm_try[pq_a] = lq_b
                        rm_try[pq_b] = lq_a
                        swapped = True
                        break
                if not swapped:
                    indices = rng.choice(len(interacting_logical),
                                         size=min(4, len(interacting_logical)),
                                         replace=False)
                    for s in range(0, len(indices) - 1, 2):
                        lq_x = interacting_logical[indices[s]]
                        lq_y = interacting_logical[indices[s + 1]]
                        pq_x, pq_y = m_try[lq_x], m_try[lq_y]
                        m_try[lq_x] = pq_y
                        m_try[lq_y] = pq_x
                        rm_try[pq_x] = lq_y
                        rm_try[pq_y] = lq_x

            elif mode == 2 and len(interacting_logical) >= 6:
                qcost = defaultdict(float)
                for (eq1, eq2), w in weights.items():
                    if m_try[eq1] >= 0 and m_try[eq2] >= 0:
                        c = w * d[m_try[eq1]][m_try[eq2]]
                        qcost[eq1] += c
                        qcost[eq2] += c
                cost_pairs = [(qcost.get(q, 0.0), q) for q in interacting_logical]
                cost_pairs.sort(reverse=True)
                top3 = [q for _, q in cost_pairs[:min(5, len(cost_pairs))]]
                if len(top3) >= 3:
                    sel = top3[:3]
                    pqs = [m_try[lq] for lq in sel]
                    m_try[sel[0]] = pqs[1]
                    m_try[sel[1]] = pqs[2]
                    m_try[sel[2]] = pqs[0]
                    rm_try[pqs[0]] = sel[2]
                    rm_try[pqs[1]] = sel[0]
                    rm_try[pqs[2]] = sel[1]
            else:
                n_swaps = min(2, len(interacting_logical) // 2)
                indices = rng.choice(len(interacting_logical),
                                     size=min(2 * n_swaps, len(interacting_logical)),
                                     replace=False)
                for s in range(0, len(indices) - 1, 2):
                    lq_a = interacting_logical[indices[s]]
                    lq_b = interacting_logical[indices[s + 1]]
                    pq_a = m_try[lq_a]
                    pq_b = m_try[lq_b]
                    m_try[lq_a] = pq_b
                    m_try[lq_b] = pq_a
                    rm_try[pq_a] = lq_b
                    rm_try[pq_b] = lq_a

            run_swap_refinement(m_try, rm_try, logical_nbrs, max_rounds=4, eff_dist=eff_dist)
            cost = compute_total_cost(m_try, weights, eff_dist)

            delta = cost - current_cost
            if delta < 0 or (T > 1e-12 and rng.random() < math.exp(-delta / T)):
                current_m = list(m_try)
                current_rm = list(rm_try)
                current_cost = cost

            if cost < best_cost:
                best_cost = cost
                best_m = list(m_try)
                best_rm = list(rm_try)

        return best_m, best_rm

    # ---------------------------------------------------------------
    # Step 7: Compute edge loads for a given mapping
    # ---------------------------------------------------------------
    def compute_edge_loads(m, weights):
        edge_load = defaultdict(float)
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                p1, p2 = m[q1], m[q2]
                edges_on_path = get_path_edges(p1, p2)
                for e in edges_on_path:
                    edge_load[e] += w
        return edge_load

    def compute_effective_distances(lambdas):
        n = len(dist)
        eff = [[0.0] * n for _ in range(n)]
        for p1 in physical_qubits:
            for p2 in physical_qubits:
                if p1 == p2:
                    eff[p1][p2] = 0.0
                else:
                    base = dist[p1][p2]
                    lag_cost = sum(lambdas.get(e, 0.0) for e in get_path_edges(p1, p2))
                    eff[p1][p2] = base + lag_cost
        return eff

    # ---------------------------------------------------------------
    # Step 8: Routing simulation for RSDIWR
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=25):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)

        if not gates_2q:
            return swap_counts

        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set()
        for g in gates_2q:
            if pred_remaining[g] == 0:
                front.add(g)

        layers_done = 0
        sim_decay = [1.0] * num_q

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
                sim_decay = [1.0] * num_q
                continue

            active_phys = set()
            for g in front:
                gq1, gq2 = gates_2q[g]
                active_phys.add(sim_m[gq1])
                active_phys.add(sim_m[gq2])

            candidates = set()
            for pq in active_phys:
                for nb in self.backend.get(pq, []):
                    candidates.add((min(pq, nb), max(pq, nb)))

            best_swap = None
            best_score = float('inf')

            for (s1, s2) in candidates:
                l1 = sim_rm[s1]
                l2 = sim_rm[s2]
                max_d = max(sim_decay[s1], sim_decay[s2])

                score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1 = sim_m[gq1]
                    p2 = sim_m[gq2]
                    if gq1 == l1:
                        p1 = s2
                    elif gq1 == l2:
                        p1 = s1
                    if gq2 == l1:
                        p2 = s2
                    elif gq2 == l2:
                        p2 = s1

                    deps = dep_count_2q.get(g, 0) + 1
                    score += deps * dist[p1][p2]

                score *= max_d
                if score < best_score:
                    best_score = score
                    best_swap = (s1, s2)

            if best_swap is None:
                break

            s1, s2 = best_swap
            l1 = sim_rm[s1]
            l2 = sim_rm[s2]
            sim_m[l1] = s2
            sim_m[l2] = s1
            sim_rm[s1] = l2
            sim_rm[s2] = l1

            sim_decay[s1] += 0.001
            sim_decay[s2] += 0.001

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0

        return swap_counts

    # ---------------------------------------------------------------
    # Step 9: Build best mapping with given weights and effective distances
    # ---------------------------------------------------------------
    def build_best_mapping(eff_weights, eff_dist=None):
        eff_nbrs, eff_deg = build_neighbors_from_weights(eff_weights)

        candidates = []
        if seed_lqs and seed_pqs:
            for s_lq in seed_lqs:
                for s_pq in seed_pqs:
                    m, rm = run_greedy_placement(s_lq, s_pq, eff_nbrs, eff_deg, eff_dist)
                    fill_unmapped(m, rm)
                    cost = compute_total_cost(m, eff_weights, eff_dist)
                    candidates.append((cost, m, rm))

        # MIS seed
        mis = build_mis_seed()
        if mis and seed_pqs:
            for s_pq in seed_pqs[:2]:
                m, rm = run_greedy_placement(mis[0], s_pq, eff_nbrs, eff_deg, eff_dist)
                fill_unmapped(m, rm)
                cost = compute_total_cost(m, eff_weights, eff_dist)
                candidates.append((cost, m, rm))

        # Spectral seed
        spec_order = build_spectral_order()
        if spec_order and len(spec_order) >= 2:
            start_pq = phys_by_centrality[0]
            bfs_order_phys = []
            visited = set()
            bfs_q = deque([start_pq])
            visited.add(start_pq)
            while bfs_q:
                cur = bfs_q.popleft()
                bfs_order_phys.append(cur)
                for nb in sorted(self.backend.get(cur, [])):
                    if nb not in visited:
                        visited.add(nb)
                        bfs_q.append(nb)

            m = [-1] * num_q
            rm = [-1] * num_q
            used_phys = set()
            for i, lq in enumerate(spec_order):
                if i < len(bfs_order_phys):
                    pq = bfs_order_phys[i]
                    m[lq] = pq
                    rm[pq] = lq
                    used_phys.add(pq)
            remaining_lq = [lq for lq in logical_qubits if m[lq] == -1]
            remaining_pq = [pq for pq in bfs_order_phys if rm[pq] == -1]
            for lq, pq in zip(remaining_lq, remaining_pq):
                m[lq] = pq
                rm[pq] = lq
            fill_unmapped(m, rm)
            cost = compute_total_cost(m, eff_weights, eff_dist)
            candidates.append((cost, m, rm))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            num_to_refine = min(4, len(candidates))

            best_cost = float('inf')
            best_m = None
            best_rm = None

            for idx in range(num_to_refine):
                _, m, rm = candidates[idx]
                m_c = list(m)
                rm_c = list(rm)

                run_swap_refinement(m_c, rm_c, eff_nbrs, max_rounds=4, eff_dist=eff_dist)
                m_c, rm_c = perturb_and_refine_sa(m_c, rm_c, eff_weights, eff_nbrs,
                                                   num_perturbations=9, eff_dist=eff_dist)

                cost = compute_total_cost(m_c, eff_weights, eff_dist)
                if cost < best_cost:
                    best_cost = cost
                    best_m = m_c
                    best_rm = rm_c

            return best_m, best_rm, best_cost
        else:
            return list(range(num_q)), list(range(num_q)), float('inf')

    # ===============================================================
    # Step 10: LAGRANGIAN RELAXATION OUTER LOOP
    #
    # Key idea: Lagrangian multipliers lambda_e for each hardware edge e.
    # Effective distance: dist(p1,p2) + sum_{e in path(p1,p2)} lambda_e
    # After each ILS/RSDIWR round, compute edge loads, update:
    #   lambda_e <- max(0, lambda_e + step*(load(e) - tau))
    # where tau = mean_load * 1.5
    # ===============================================================
    LAGRANGIAN_ITERS = 3
    RSDIWR_ITERS = 4
    alpha_schedule = [1.0, 0.65, 0.35, 0.15]

    lambdas = defaultdict(float)

    best_overall_m = None
    best_overall_rm = None
    best_overall_cost = float('inf')

    swap_counts = defaultdict(float)

    for lag_iter in range(LAGRANGIAN_ITERS):
        # Compute effective distances with current Lagrangian multipliers
        eff_dist = compute_effective_distances(lambdas)

        # RSDIWR loop with Lagrangian-adjusted distances
        for t in range(RSDIWR_ITERS):
            alpha_blend = alpha_schedule[t]

            eff_weights = defaultdict(float)
            if t == 0 or not swap_counts:
                for key, w in static_weight.items():
                    eff_weights[key] = w
            else:
                max_s = max(static_weight.values()) if static_weight else 1.0
                max_sw = max(swap_counts.values()) if swap_counts else 1.0
                scale = max_s / max(max_sw, 1e-10)
                a_keys = set(static_weight.keys()) | set(swap_counts.keys())
                for key in a_keys:
                    w_s = static_weight.get(key, 0.0)
                    w_r = swap_counts.get(key, 0.0) * scale
                    eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

            cur_m, cur_rm, _ = build_best_mapping(eff_weights, eff_dist)

            # Evaluate on base distances (true cost)
            eval_cost = compute_total_cost(cur_m, static_weight)
            if eval_cost < best_overall_cost:
                best_overall_cost = eval_cost
                best_overall_m = list(cur_m)
                best_overall_rm = list(cur_rm)

            if t < RSDIWR_ITERS - 1:
                routing_layers = 14 + 7 * t
                swap_counts = simulate_routing(cur_m, cur_rm, max_layers=routing_layers)

        # Lagrangian multiplier update (subgradient ascent)
        if lag_iter < LAGRANGIAN_ITERS - 1:
            edge_loads = compute_edge_loads(best_overall_m, static_weight)

            if edge_loads:
                mean_load = sum(edge_loads.values()) / max(len(hw_edges), 1)
                tau = mean_load * 1.5

                step_size = 0.1 / (lag_iter + 1)

                for e in hw_edges:
                    load_e = edge_loads.get(e, 0.0)
                    subgradient = load_e - tau
                    lambdas[e] = max(0.0, lambdas[e] + step_size * subgradient)

            swap_counts = defaultdict(float)

    # ---------------------------------------------------------------
    # Step 11: Set final mapping
    # ---------------------------------------------------------------
    if best_overall_m is None:
        best_overall_m = list(range(num_q))
        best_overall_rm = list(range(num_q))

    self.mapping_dict = best_overall_m
    self.reverse_mapping_dict = best_overall_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)