def init_mapping(self):
    """
    Optimal Transport (Wasserstein) Placement.

    Uses optimal transport theory to map logical qubits to physical qubits.
    Builds a transport-inspired cost matrix from logical interaction profiles
    and hardware distance profiles, then solves via Hungarian algorithm.
    Refines with 2-opt local search on QAP cost.
    """
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import math

    gates_list = list(self.access.items())

    # Collect logical qubits
    logical_qubit_set = set()
    for _, qubits in gates_list:
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_lq = len(logical_qubits)
    n_pq = len(physical_qubits)

    # --- Step 1: Build logical interaction graph with layer-weighted edges ---
    qubit_ready = {}
    gate_layers = []
    for _, qubits in gates_list:
        es = max((qubit_ready.get(q, 0) for q in qubits), default=0)
        gate_layers.append(es)
        for q in qubits:
            qubit_ready[q] = es + 1

    total_depth = max((qubit_ready.get(q, 0) for q in logical_qubit_set), default=1)
    half_life = max(total_depth / 4.0, 3.0)

    interaction_weight = defaultdict(float)
    for idx, (_, qubits) in enumerate(gates_list):
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_layers[idx]
            w = math.exp(-layer * math.log(2) / half_life) + 0.05
            interaction_weight[key] += w

    # Build adjacency for logical interaction graph
    lq_adj = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        lq_adj[q1][q2] = w
        lq_adj[q2][q1] = w

    # --- Step 2: All-pairs shortest paths on logical interaction graph (Floyd-Warshall) ---
    lq_index = {q: i for i, q in enumerate(logical_qubits)}

    logical_dist = np.full((n_lq, n_lq), float('inf'))
    for i in range(n_lq):
        logical_dist[i][i] = 0.0

    for (q1, q2) in interaction_weight:
        i, j = lq_index[q1], lq_index[q2]
        logical_dist[i][j] = 1.0
        logical_dist[j][i] = 1.0

    for k in range(n_lq):
        for i in range(n_lq):
            if logical_dist[i][k] == float('inf'):
                continue
            for j in range(n_lq):
                d = logical_dist[i][k] + logical_dist[k][j]
                if d < logical_dist[i][j]:
                    logical_dist[i][j] = d

    # --- Step 3: Interaction weights per qubit ---
    qubit_total_weight = np.zeros(n_lq)
    for (q1, q2), w in interaction_weight.items():
        qubit_total_weight[lq_index[q1]] += w
        qubit_total_weight[lq_index[q2]] += w

    # --- Step 4: Hardware distance matrix ---
    pq_index = {q: i for i, q in enumerate(physical_qubits)}
    hw_dist = np.zeros((n_pq, n_pq))
    for i, pq1 in enumerate(physical_qubits):
        for j, pq2 in enumerate(physical_qubits):
            hw_dist[i][j] = self.distance_matrix[pq1][pq2]

    # --- Step 5: Transport-inspired cost matrix ---
    # Logical demand profiles (interaction-weighted proximity)
    logical_profiles = np.zeros((n_lq, n_lq))
    for i in range(n_lq):
        for k in range(n_lq):
            if i != k and logical_dist[i][k] < float('inf'):
                logical_profiles[i][k] = qubit_total_weight[k] / (1.0 + logical_dist[i][k])

    # Hardware proximity profiles
    hw_profiles = np.zeros((n_pq, n_pq))
    for j in range(n_pq):
        for l in range(n_pq):
            if j != l:
                hw_profiles[j][l] = 1.0 / (1.0 + hw_dist[j][l])

    cost_matrix = np.zeros((n_lq, n_pq))
    max_len = max(n_lq, n_pq)

    for i in range(n_lq):
        # Sorted demand profile
        demands = sorted(logical_profiles[i], reverse=True)
        while len(demands) < max_len:
            demands.append(0.0)
        demands = demands[:max_len]
        demand_arr = np.array(demands)
        demand_sum = demand_arr.sum()
        if demand_sum > 0:
            demand_arr = demand_arr / demand_sum

        for j in range(n_pq):
            # Sorted supply profile
            supplies = sorted(hw_profiles[j], reverse=True)
            while len(supplies) < max_len:
                supplies.append(0.0)
            supplies = supplies[:max_len]
            supply_arr = np.array(supplies)
            supply_sum = supply_arr.sum()
            if supply_sum > 0:
                supply_arr = supply_arr / supply_sum

            # 1-Wasserstein on sorted 1D distributions (EMD)
            cost_matrix[i][j] = np.sum(np.abs(np.cumsum(demand_arr) - np.cumsum(supply_arr)))

    # Direct interaction-distance cost component
    direct_cost = np.zeros((n_lq, n_pq))
    for i in range(n_lq):
        li = logical_qubits[i]
        neighbors = lq_adj.get(li, {})
        if not neighbors:
            continue
        for k_logical, w_ik in neighbors.items():
            k = lq_index.get(k_logical)
            if k is None:
                continue
            for j in range(n_pq):
                best_dist = np.partition(hw_dist[j], min(3, n_pq - 1))[:min(3, n_pq)].mean()
                direct_cost[i][j] += w_ik * best_dist

    # Normalize and combine
    ws_max = cost_matrix.max()
    if ws_max > 0:
        cost_matrix = cost_matrix / ws_max
    dc_max = direct_cost.max()
    if dc_max > 0:
        direct_cost = direct_cost / dc_max

    combined_cost = 0.6 * cost_matrix + 0.4 * direct_cost

    # --- Step 6: Hungarian algorithm ---
    if n_lq < n_pq:
        padded_cost = np.zeros((n_pq, n_pq))
        padded_cost[:n_lq, :] = combined_cost
        row_ind, col_ind = linear_sum_assignment(padded_cost)
    else:
        row_ind, col_ind = linear_sum_assignment(combined_cost)

    assignment = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_lq:
            assignment[logical_qubits[r]] = physical_qubits[c]

    # --- Step 7: QAP cost ---
    def qap_cost(assign):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            if q1 in assign and q2 in assign:
                cost += w * self.distance_matrix[assign[q1]][assign[q2]]
        return cost

    # --- Step 8: 2-opt local search ---
    best_assign = dict(assignment)
    best_cost = qap_cost(best_assign)

    used_physical = set(best_assign.values())
    free_physical = [p for p in physical_qubits if p not in used_physical]

    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(n_lq):
            for j in range(i + 1, n_lq):
                li, lj = logical_qubits[i], logical_qubits[j]
                pi, pj = best_assign[li], best_assign[lj]

                delta = 0.0
                for k_logical, w in lq_adj.get(li, {}).items():
                    if k_logical == lj:
                        continue
                    if k_logical in best_assign:
                        pk = best_assign[k_logical]
                        delta += w * (self.distance_matrix[pj][pk] - self.distance_matrix[pi][pk])
                for k_logical, w in lq_adj.get(lj, {}).items():
                    if k_logical == li:
                        continue
                    if k_logical in best_assign:
                        pk = best_assign[k_logical]
                        delta += w * (self.distance_matrix[pi][pk] - self.distance_matrix[pj][pk])

                if delta < -1e-9:
                    best_assign[li] = pj
                    best_assign[lj] = pi
                    best_cost += delta
                    improved = True

        for i in range(n_lq):
            li = logical_qubits[i]
            pi = best_assign[li]
            for fp in free_physical:
                delta = 0.0
                for k_logical, w in lq_adj.get(li, {}).items():
                    if k_logical in best_assign:
                        pk = best_assign[k_logical]
                        delta += w * (self.distance_matrix[fp][pk] - self.distance_matrix[pi][pk])
                if delta < -1e-9:
                    best_assign[li] = fp
                    free_physical.remove(fp)
                    free_physical.append(pi)
                    best_cost += delta
                    improved = True
                    break

    # --- Step 9: Build final mapping ---
    self.mapping_dict = list(range(self.num_qubits))
    self.reverse_mapping_dict = list(range(self.num_qubits))

    used_physical = set(best_assign.values())
    available_physical = [p for p in physical_qubits if p not in used_physical]

    all_logical_needed = set(range(self.num_qubits))
    mapped_logical = set(best_assign.keys())
    unmapped_logical = sorted(all_logical_needed - mapped_logical)

    avail_idx = 0
    full_assign = dict(best_assign)
    for lq in unmapped_logical:
        if avail_idx < len(available_physical):
            full_assign[lq] = available_physical[avail_idx]
            avail_idx += 1

    all_assigned_physical = set(full_assign.values())
    remaining_physical = sorted(set(range(self.num_qubits)) - all_assigned_physical)
    remaining_logical = sorted(set(range(self.num_qubits)) - set(full_assign.keys()))
    for lq, pq in zip(remaining_logical, remaining_physical):
        full_assign[lq] = pq

    for lq in range(self.num_qubits):
        pq = full_assign[lq]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)