def init_mapping(self):
    """
    Recursive Spectral Bisection Placement.

    Simultaneously and recursively bisects both the logical interaction graph
    and the hardware coupling graph using the Fiedler vector. At each level,
    the denser logical half is matched to the denser hardware half. Recursion
    terminates at singletons, producing a hierarchical bijective assignment.
    Refined with 2-opt local search.
    """
    import numpy as np
    from collections import defaultdict
    import math

    # --- Extract logical qubits and build interaction graph ---
    logical_qubit_set = set()
    interaction_weight = defaultdict(float)

    gates_list = list(self.access.items())

    # Forward pass for layer info (used for temporal weighting)
    qubit_ready = {}
    gate_layer = []
    for _, qubits in gates_list:
        es = max((qubit_ready.get(q, 0) for q in qubits), default=0)
        gate_layer.append(es)
        for q in qubits:
            qubit_ready[q] = es + 1

    total_depth = max((qubit_ready[q] for q in qubit_ready), default=1)
    half_life = max(total_depth / 4.0, 4.0)

    for idx, (_, qubits) in enumerate(gates_list):
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            layer = gate_layer[idx]
            w = math.exp(-layer * math.log(2) / half_life) + 0.05
            interaction_weight[key] += w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # --- Handle trivial / empty cases ---
    if not logical_qubits or not interaction_weight:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            from src.mapping.mapping import dict_to_isl_map
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n_logical = len(logical_qubits)
    n_physical = len(physical_qubits)

    # --- Helper functions ---
    def build_laplacian(nodes, edges_weights):
        """Build weighted Laplacian matrix for a set of nodes given edge weights."""
        n = len(nodes)
        if n <= 1:
            return np.zeros((n, n)), nodes
        node_idx = {v: i for i, v in enumerate(nodes)}
        L = np.zeros((n, n))
        for (u, v), w in edges_weights.items():
            if u in node_idx and v in node_idx:
                i, j = node_idx[u], node_idx[v]
                L[i, j] -= w
                L[j, i] -= w
                L[i, i] += w
                L[j, j] += w
        return L, nodes

    def build_hardware_laplacian(nodes):
        """Build unweighted Laplacian for a subset of the hardware graph."""
        n = len(nodes)
        if n <= 1:
            return np.zeros((n, n))
        node_set = set(nodes)
        node_idx = {v: i for i, v in enumerate(nodes)}
        L = np.zeros((n, n))
        for u in nodes:
            for v in self.backend.get(u, set()):
                if v in node_set:
                    i, j = node_idx[u], node_idx[v]
                    if i < j:
                        L[i, j] -= 1.0
                        L[j, i] -= 1.0
                        L[i, i] += 1.0
                        L[j, j] += 1.0
        return L

    def fiedler_vector(L):
        """Compute the Fiedler vector (eigenvector of second smallest eigenvalue)."""
        n = L.shape[0]
        if n <= 1:
            return np.zeros(n)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            return eigenvectors[:, 1]
        except np.linalg.LinAlgError:
            return np.zeros(n)

    def partition_by_fiedler(nodes, fiedler_vals):
        """Split nodes into two halves sorted by Fiedler value."""
        indexed = sorted(zip(fiedler_vals, range(len(nodes))))
        mid = len(nodes) // 2
        part_a = [nodes[indexed[i][1]] for i in range(mid)]
        part_b = [nodes[indexed[i][1]] for i in range(mid, len(nodes))]
        return part_a, part_b

    def internal_weight_logical(nodes):
        """Sum of interaction weights among a set of logical qubits."""
        node_set = set(nodes)
        total = 0.0
        for (u, v), w in interaction_weight.items():
            if u in node_set and v in node_set:
                total += w
        return total

    def internal_connectivity_hardware(nodes):
        """Count of edges among a set of physical qubits."""
        node_set = set(nodes)
        count = 0
        for u in nodes:
            for v in self.backend.get(u, set()):
                if v in node_set and u < v:
                    count += 1
        return count

    # --- Recursive Spectral Bisection ---
    def recursive_bisection(log_nodes, phys_nodes):
        n = len(log_nodes)

        if n == 0:
            return {}
        if n == 1:
            return {log_nodes[0]: phys_nodes[0]}
        if n == 2:
            if len(phys_nodes) >= 2:
                key = (min(log_nodes[0], log_nodes[1]), max(log_nodes[0], log_nodes[1]))
                w = interaction_weight.get(key, 0)
                if w > 0:
                    cost_a = w * self.distance_matrix[phys_nodes[0]][phys_nodes[1]]
                    cost_b = w * self.distance_matrix[phys_nodes[1]][phys_nodes[0]]
                    if cost_b < cost_a:
                        return {log_nodes[0]: phys_nodes[1], log_nodes[1]: phys_nodes[0]}
                return {log_nodes[0]: phys_nodes[0], log_nodes[1]: phys_nodes[1]}
            else:
                return {log_nodes[0]: phys_nodes[0]}

        phys_nodes = phys_nodes[:n]

        # Compute Fiedler vectors for both graphs
        log_set = set(log_nodes)
        local_edges = {(u, v): w for (u, v), w in interaction_weight.items()
                       if u in log_set and v in log_set}

        L_log, _ = build_laplacian(log_nodes, local_edges)
        fiedler_log = fiedler_vector(L_log)

        L_hw = build_hardware_laplacian(phys_nodes)
        fiedler_hw = fiedler_vector(L_hw)

        # Bisect logical qubits
        log_a, log_b = partition_by_fiedler(log_nodes, fiedler_log)

        # Partition physical qubits to match logical partition sizes
        target_a = len(log_a)
        target_b = len(log_b)
        phys_indexed = sorted(zip(fiedler_hw, range(len(phys_nodes))))
        phys_sorted = [phys_nodes[phys_indexed[i][1]] for i in range(len(phys_nodes))]
        phys_a = phys_sorted[:target_a]
        phys_b = phys_sorted[target_a:target_a + target_b]

        # Match denser logical half -> denser hardware half
        log_w_a = internal_weight_logical(log_a)
        log_w_b = internal_weight_logical(log_b)
        hw_c_a = internal_connectivity_hardware(phys_a)
        hw_c_b = internal_connectivity_hardware(phys_b)

        if (log_w_a >= log_w_b) != (hw_c_a >= hw_c_b):
            phys_a, phys_b = phys_b, phys_a

        result = {}
        result.update(recursive_bisection(log_a, phys_a))
        result.update(recursive_bisection(log_b, phys_b))
        return result

    # --- Select physical qubits (pick highest-degree subset if needed) ---
    if n_physical > n_logical:
        degree = {p: len(self.backend.get(p, set())) for p in physical_qubits}
        phys_candidates = sorted(physical_qubits, key=lambda p: -degree[p])[:n_logical]
        phys_candidates.sort()
    else:
        phys_candidates = physical_qubits[:n_logical]

    lq_phys = recursive_bisection(logical_qubits, phys_candidates)

    # --- 2-opt local search refinement ---
    best_cost = sum(
        w * self.distance_matrix[lq_phys[q1]][lq_phys[q2]]
        for (q1, q2), w in interaction_weight.items()
        if q1 in lq_phys and q2 in lq_phys
    )
    improved = True
    max_iters = 50
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                lq_i = logical_qubits[i]
                lq_j = logical_qubits[j]
                p_i = lq_phys[lq_i]
                p_j = lq_phys[lq_j]

                delta = 0.0
                for (q1, q2), w in interaction_weight.items():
                    if q1 not in lq_phys or q2 not in lq_phys:
                        continue
                    touches_i = (q1 == lq_i or q2 == lq_i)
                    touches_j = (q1 == lq_j or q2 == lq_j)
                    if not touches_i and not touches_j:
                        continue
                    old_p1 = lq_phys[q1]
                    old_p2 = lq_phys[q2]
                    new_p1 = p_j if q1 == lq_i else (p_i if q1 == lq_j else old_p1)
                    new_p2 = p_j if q2 == lq_i else (p_i if q2 == lq_j else old_p2)
                    delta += w * (self.distance_matrix[new_p1][new_p2] -
                                  self.distance_matrix[old_p1][old_p2])

                if delta < -1e-12:
                    lq_phys[lq_i] = p_j
                    lq_phys[lq_j] = p_i
                    best_cost += delta
                    improved = True

    # --- Commit assignment via in-place swaps (guarantees bijection) ---
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_phys.items():
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
        from src.mapping.mapping import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)