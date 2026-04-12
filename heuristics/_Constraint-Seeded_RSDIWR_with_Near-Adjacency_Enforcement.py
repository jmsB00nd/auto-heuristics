def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    import math

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # ---------------------------------------------------------------
    # Step 1: Build precise read/write DAG + topological sort + critical path
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

    # Kahn's topological sort + gate layers
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

    # Critical path: number of successor layers remaining
    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # ---------------------------------------------------------------
    # Step 2: Build dual-signal weight schemes
    # ---------------------------------------------------------------
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha_decay = 2.5

    logical_qubits_set = set()

    # Scheme A: critical-path weighted
    static_weight_A = defaultdict(float)
    logical_degree_A = defaultdict(float)

    # Scheme B: geometric mean of cp and temporal decay
    static_weight_B = defaultdict(float)
    logical_degree_B = defaultdict(float)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        logical_qubits_set.add(q1)
        logical_qubits_set.add(q2)
        key = (min(q1, q2), max(q1, q2))

        cp = critical_path[g] + 1
        layer = gate_layer[g]
        w_cp = cp * (max_layer - layer + 1)

        static_weight_A[key] += w_cp
        logical_degree_A[q1] += w_cp
        logical_degree_A[q2] += w_cp

        r = topo_rank.get(g, 0)
        w_td = math.exp(-alpha_decay * r / total_gates)
        w_geo = math.sqrt(w_cp * w_td)

        static_weight_B[key] += w_geo
        logical_degree_B[q1] += w_geo
        logical_degree_B[q2] += w_geo

    for g in all_gates:
        if len(self.access[g]) == 1:
            logical_qubits_set.add(self.access[g][0])

    logical_qubits = sorted(logical_qubits_set)

    # Combined degree for seed selection
    logical_degree_combined = defaultdict(float)
    for q in logical_qubits:
        logical_degree_combined[q] = logical_degree_A.get(q, 0) + logical_degree_B.get(q, 0)

    interacting_logical = [q for q in logical_qubits if logical_degree_combined.get(q, 0) > 0]

    # ---------------------------------------------------------------
    # Step 3: Precompute physical graph properties
    # ---------------------------------------------------------------
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in physical_qubits)

    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            hw_adj[pq].add(pq2)

    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}

    # Multi-seed selection
    if interacting_logical:
        sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree_combined[q], reverse=True)
        seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    else:
        seed_lqs = logical_qubits[:1] if logical_qubits else []

    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    # Degree ranks for connectivity-matching tie-breaking
    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree_combined[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)

    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    # ---------------------------------------------------------------
    # Step 4: Build 2q-only DAG for routing simulation
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
    q2 = deque(sorted(g for g in gates_2q if in_deg_2q[g] == 0))
    while q2:
        g = q2.popleft()
        topo_2q.append(g)
        for s in dag2q_succ[g]:
            in_deg_2q[s] -= 1
            if in_deg_2q[s] == 0:
                q2.append(s)

    dep_count_2q = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count_2q[g] += dep_count_2q[s] + 1

    # ---------------------------------------------------------------
    # Step 5: CSP — Constraint-Seeded Partial Assignment via AC-3
    # ---------------------------------------------------------------
    # Compute dual-signal combined pair weights for CSP ranking
    dual_pair_weight = defaultdict(float)
    max_A_val = max(static_weight_A.values()) if static_weight_A else 1.0
    max_B_val = max(static_weight_B.values()) if static_weight_B else 1.0
    for key in set(static_weight_A.keys()) | set(static_weight_B.keys()):
        w_a = static_weight_A.get(key, 0.0) / max(max_A_val, 1e-10)
        w_b = static_weight_B.get(key, 0.0) / max(max_B_val, 1e-10)
        dual_pair_weight[key] = w_a + w_b

    n_logical = len(logical_qubits)
    K = min(15, max(1, n_logical // 3))

    # Sort pairs by dual-signal weight descending
    sorted_pairs = sorted(dual_pair_weight.items(), key=lambda x: -x[1])
    top_k_pairs = sorted_pairs[:K]

    # Precompute: for each physical qubit, set of physicals within distance <= 2
    within_dist2 = {}
    for p in physical_qubits:
        within_dist2[p] = set()
        for p2 in physical_qubits:
            if p != p2 and self.distance_matrix[p][p2] <= 2:
                within_dist2[p].add(p2)

    def solve_csp(pair_list):
        """AC-3 + backtracking CSP solver for near-adjacency constraints."""
        if not pair_list:
            return {}

        # Collect CSP variables (logical qubits in constraints)
        csp_vars = set()
        for (lq1, lq2), _ in pair_list:
            csp_vars.add(lq1)
            csp_vars.add(lq2)
        csp_vars = sorted(csp_vars)

        # Initialize domains
        domains = {v: set(physical_qubits) for v in csp_vars}

        # Constraint edges
        constraint_edges = [(lq1, lq2) for (lq1, lq2), _ in pair_list]

        # AC-3: arc consistency
        arc_queue = deque()
        for (a, b) in constraint_edges:
            arc_queue.append((a, b))
            arc_queue.append((b, a))

        def revise(xi, xj):
            revised = False
            to_remove = []
            for vi in domains[xi]:
                # Need at least one vj in domains[xj] with vj != vi and dist(vi,vj) <= 2
                has_support = False
                for vj in domains[xj]:
                    if vj != vi and self.distance_matrix[vi][vj] <= 2:
                        has_support = True
                        break
                if not has_support:
                    to_remove.append(vi)
                    revised = True
            for vi in to_remove:
                domains[xi].discard(vi)
            return revised

        while arc_queue:
            xi, xj = arc_queue.popleft()
            if xi not in domains or xj not in domains:
                continue
            if revise(xi, xj):
                if len(domains[xi]) == 0:
                    return None  # Infeasible
                for (a, b) in constraint_edges:
                    if b == xi and a != xj:
                        arc_queue.append((a, xi))
                    elif a == xi and b != xj:
                        arc_queue.append((b, xi))

        # Backtracking with fail-first (MRV) variable ordering
        def backtrack(assignment, used_phys, remaining):
            if not remaining:
                return dict(assignment)

            # MRV: choose variable with smallest remaining domain
            var = min(remaining, key=lambda v: len(domains[v]))
            if len(domains[var]) == 0:
                return None

            # Sort candidates by number of constraints they satisfy (prefer central physicals)
            candidates = [v for v in sorted(domains[var], key=lambda p: -len(within_dist2[p])) if v not in used_phys]

            for val in candidates:
                assignment[var] = val
                used_phys.add(val)

                # Forward checking: save and prune domains
                saved = {}
                feasible = True

                # Constraint pruning: partners must be within dist 2
                for (a, b) in constraint_edges:
                    other = None
                    if a == var:
                        other = b
                    elif b == var:
                        other = a
                    if other is not None and other in remaining and other != var:
                        if other not in saved:
                            saved[other] = set(domains[other])
                        domains[other] = {v for v in domains[other]
                                          if v != val and self.distance_matrix[val][v] <= 2}
                        if len(domains[other]) == 0:
                            feasible = False
                            break

                # All-different pruning
                if feasible:
                    for v in remaining:
                        if v != var and v not in saved:
                            if val in domains[v]:
                                saved[v] = set(domains[v])
                                domains[v] = domains[v] - {val}
                        elif v != var and val in domains[v]:
                            domains[v] = domains[v] - {val}

                if feasible:
                    new_remaining = [v for v in remaining if v != var]
                    result = backtrack(assignment, used_phys, new_remaining)
                    if result is not None:
                        return result

                # Undo
                del assignment[var]
                used_phys.discard(val)
                for v, d in saved.items():
                    domains[v] = d

            return None

        remaining = sorted(csp_vars, key=lambda v: len(domains[v]))
        return backtrack({}, set(), remaining)

    # Try CSP with progressive relaxation
    csp_assignment = None
    current_pairs = list(top_k_pairs)

    while current_pairs:
        csp_assignment = solve_csp(current_pairs)
        if csp_assignment is not None:
            break
        # Relax: remove lowest-weight pair and retry
        current_pairs.pop()

    if csp_assignment is None:
        csp_assignment = {}

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

    def run_greedy_placement(start_lq, start_pq, logical_nbrs, eff_degree):
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
                        dist = self.distance_matrix[pq][m[plq]]
                        cost = iw * dist
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

    def run_csp_greedy_fill(csp_partial, logical_nbrs, eff_degree):
        """Build a full mapping from the CSP partial assignment using greedy BFS fill."""
        m = [-1] * num_q
        rm = [-1] * num_q
        used_phys = set()
        placed = set()

        # Pin CSP-assigned qubits
        for lq, pq in csp_partial.items():
            m[lq] = pq
            rm[pq] = lq
            used_phys.add(pq)
            placed.add(lq)

        # Greedy fill remaining logical qubits by interaction affinity
        remaining = [q for q in logical_qubits if q not in placed]
        # Sort by total interaction weight with already-placed qubits (descending)
        remaining.sort(key=lambda lq: -sum(logical_nbrs[lq].get(plq, 0.0) for plq in placed))

        max_iw = max((w for nbrs in logical_nbrs.values() for w in nbrs.values()), default=1.0)

        for lq in remaining:
            nbrs_placed = {plq: logical_nbrs[lq].get(plq, 0.0)
                           for plq in placed if plq in logical_nbrs[lq]}

            if nbrs_placed:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        dist = self.distance_matrix[pq][m[plq]]
                        cost = iw * dist
                        # Adjacency bonus
                        if m[plq] in hw_adj[pq]:
                            cost *= (0.90 - 0.10 * (iw / max_iw))
                        score += cost
                    if score < best_score:
                        best_score = score
                        best_pq = pq
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_phys:
                        score = phys_centrality[pq]
                        if score < best_score:
                            best_score = score
                            best_pq = pq

            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                used_phys.add(best_pq)
                placed.add(lq)

        return m, rm

    def fill_unmapped(m, rm):
        unmapped = [q for q in range(num_q) if m[q] == -1]
        free = [pq for pq in range(num_q) if rm[pq] == -1]
        for lq, pq in zip(unmapped, free):
            m[lq] = pq
            rm[pq] = lq

    def compute_total_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    def run_swap_refinement(m, rm, logical_nbrs, max_rounds):
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
                            delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])
                        w_b = logical_nbrs[lq_b].get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])

                    if delta < -1e-12:
                        m[lq_a] = pq_b
                        m[lq_b] = pq_a
                        rm[pq_a] = lq_b
                        rm[pq_b] = lq_a
                        improved = True

    def perturb_and_refine_sa(m, rm, weights, logical_nbrs, num_perturbations=9):
        """SA-based perturbation with 4 modes including 3-opt chain."""
        if len(interacting_logical) < 4:
            return m, rm
        best_m = list(m)
        best_rm = list(rm)
        best_cost = compute_total_cost(m, weights)
        current_m = list(m)
        current_rm = list(rm)
        current_cost = best_cost

        rng_seed = int(best_cost * 1000) % (2**31)
        rng = np.random.RandomState(rng_seed)

        T_init = max(best_cost * 0.02, 1.0)
        T_min = T_init * 0.01

        def compute_edge_costs(mc):
            edge_costs = []
            for (eq1, eq2), w in weights.items():
                if mc[eq1] >= 0 and mc[eq2] >= 0:
                    c = w * self.distance_matrix[mc[eq1]][mc[eq2]]
                    edge_costs.append((c, eq1, eq2))
            edge_costs.sort(reverse=True)
            return edge_costs

        for p_idx in range(num_perturbations):
            m_try = list(current_m)
            rm_try = list(current_rm)
            mode = p_idx % 4

            T = T_init * (T_min / T_init) ** (p_idx / max(num_perturbations - 1, 1))

            if mode == 0 and len(interacting_logical) >= 4:
                edge_costs = compute_edge_costs(m_try)
                for _, eq1, eq2 in edge_costs[:min(3, len(edge_costs))]:
                    pq1 = m_try[eq1]
                    pq2 = m_try[eq2]
                    current_dist = self.distance_matrix[pq1][pq2]
                    best_delta = 0.0
                    best_swap_pair = None
                    for adj_pq in hw_adj[pq1]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq2:
                            continue
                        new_dist = self.distance_matrix[adj_pq][pq2]
                        delta = new_dist - current_dist
                        if delta < best_delta:
                            best_delta = delta
                            best_swap_pair = (eq1, occ, pq1, adj_pq)
                    for adj_pq in hw_adj[pq2]:
                        occ = rm_try[adj_pq]
                        if occ < 0 or occ == eq1:
                            continue
                        new_dist = self.distance_matrix[pq1][adj_pq]
                        delta = new_dist - current_dist
                        if delta < best_delta:
                            best_delta = delta
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
                        c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
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
                        c = w * self.distance_matrix[m_try[eq1]][m_try[eq2]]
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

            run_swap_refinement(m_try, rm_try, logical_nbrs, max_rounds=4)
            cost = compute_total_cost(m_try, weights)

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
    # Step 7: Routing simulation (25 layers)
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
                    score += deps * self.distance_matrix[p1][p2]

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
    # Step 8: Build best mapping with effective weights (+ CSP candidate)
    # ---------------------------------------------------------------
    def build_best_mapping(eff_weights, include_csp=True):
        eff_nbrs, eff_deg = build_neighbors_from_weights(eff_weights)

        candidates = []

        # Standard greedy seed candidates
        if seed_lqs and seed_pqs:
            for s_lq in seed_lqs:
                for s_pq in seed_pqs:
                    m, rm = run_greedy_placement(s_lq, s_pq, eff_nbrs, eff_deg)
                    fill_unmapped(m, rm)
                    cost = compute_total_cost(m, eff_weights)
                    candidates.append((cost, m, rm))

        # CSP-seeded candidate: the novel addition
        if include_csp and csp_assignment:
            m_csp, rm_csp = run_csp_greedy_fill(csp_assignment, eff_nbrs, eff_deg)
            fill_unmapped(m_csp, rm_csp)
            cost_csp = compute_total_cost(m_csp, eff_weights)
            candidates.append((cost_csp, m_csp, rm_csp))

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

                run_swap_refinement(m_c, rm_c, eff_nbrs, max_rounds=4)
                m_c, rm_c = perturb_and_refine_sa(m_c, rm_c, eff_weights, eff_nbrs, num_perturbations=9)

                cost = compute_total_cost(m_c, eff_weights)
                if cost < best_cost:
                    best_cost = cost
                    best_m = m_c
                    best_rm = rm_c

            return best_m, best_rm, best_cost
        else:
            return list(range(num_q)), list(range(num_q)), float('inf')

    # ---------------------------------------------------------------
    # Step 9: RSDIWR outer loop with competitive dual-weight schemes
    # ---------------------------------------------------------------
    T_iters = 4
    alpha_schedule = [1.0, 0.65, 0.35, 0.15]
    swap_counts = defaultdict(float)

    best_overall_m = None
    best_overall_rm = None
    best_overall_cost = float('inf')

    # Unified evaluation weight: average of both schemes (normalized)
    max_A = max(static_weight_A.values()) if static_weight_A else 1.0
    max_B = max(static_weight_B.values()) if static_weight_B else 1.0
    static_weight_eval = defaultdict(float)
    all_static_keys = set(static_weight_A.keys()) | set(static_weight_B.keys())
    for key in all_static_keys:
        w_a = static_weight_A.get(key, 0.0) / max(max_A, 1e-10)
        w_b = static_weight_B.get(key, 0.0) / max(max_B, 1e-10)
        static_weight_eval[key] = (w_a + w_b) * 0.5

    for t in range(T_iters):
        alpha_blend = alpha_schedule[t]

        best_iter_m = None
        best_iter_rm = None
        best_iter_eval_cost = float('inf')

        for scheme_idx, static_weight in enumerate([static_weight_A, static_weight_B]):
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

            # Include CSP candidate on first iteration, standard thereafter
            cur_m, cur_rm, _ = build_best_mapping(eff_weights, include_csp=(t == 0))

            eval_cost = compute_total_cost(cur_m, static_weight_eval)
            if eval_cost < best_iter_eval_cost:
                best_iter_eval_cost = eval_cost
                best_iter_m = cur_m
                best_iter_rm = cur_rm

        if best_iter_eval_cost < best_overall_cost:
            best_overall_cost = best_iter_eval_cost
            best_overall_m = list(best_iter_m)
            best_overall_rm = list(best_iter_rm)

        if t < T_iters - 1:
            swap_counts = simulate_routing(best_iter_m, best_iter_rm, max_layers=25)

    # ---------------------------------------------------------------
    # Step 10: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_overall_m
    self.reverse_mapping_dict = best_overall_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)