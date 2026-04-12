def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    # --- Step 1: Build weighted interaction graph from circuit gates ---
    interaction_weight = defaultdict(float)
    logical_qubit_set = set()

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback to trivial identity mapping if no logical qubits
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Helper: Fiedler vector for weighted logical interaction subgraph ---
    def fiedler_logical(nodes):
        n = len(nodes)
        if n == 1:
            return {nodes[0]: 0.0}
        node_idx = {node: i for i, node in enumerate(nodes)}
        node_set = set(nodes)
        L = np.zeros((n, n))
        for (q1, q2), w in interaction_weight.items():
            if q1 in node_set and q2 in node_set:
                i, j = node_idx[q1], node_idx[q2]
                L[i][j] -= w
                L[j][i] -= w
                L[i][i] += w
                L[j][j] += w
        try:
            eigvals, vecs = np.linalg.eigh(L)
            fvec = vecs[:, 1]
        except Exception:
            fvec = np.arange(n, dtype=float)
        return {nodes[i]: float(fvec[i]) for i in range(n)}

    # --- Helper: Fiedler vector for hardware coupling subgraph ---
    def fiedler_hardware(hw_nodes):
        n = len(hw_nodes)
        if n == 1:
            return {hw_nodes[0]: 0.0}
        node_idx = {node: i for i, node in enumerate(hw_nodes)}
        node_set = set(hw_nodes)
        L = np.zeros((n, n))
        for u in hw_nodes:
            for v in self.backend.get(u, []):
                if v in node_set:
                    i, j = node_idx[u], node_idx[v]
                    if i < j:
                        L[i][j] -= 1.0
                        L[j][i] -= 1.0
                        L[i][i] += 1.0
                        L[j][j] += 1.0
        try:
            eigvals, vecs = np.linalg.eigh(L)
            fvec = vecs[:, 1]
        except Exception:
            fvec = np.arange(n, dtype=float)
        return {hw_nodes[i]: float(fvec[i]) for i in range(n)}

    # --- Helper: bisect nodes into two halves by Fiedler ordering ---
    def bisect(nodes, fiedler_dict):
        if len(nodes) <= 1:
            return nodes[:], []
        sorted_nodes = sorted(nodes, key=lambda n: fiedler_dict[n])
        mid = len(sorted_nodes) // 2
        if mid == 0:
            mid = 1
        return sorted_nodes[:mid], sorted_nodes[mid:]

    # --- Helper: compute cross-partition interaction cost for an assignment ---
    def cross_partition_cost(l_part1, l_part2, p_part1, p_part2):
        """Evaluate cost of assigning l_part1->p_part1, l_part2->p_part2."""
        if not l_part1 or not l_part2 or not p_part1 or not p_part2:
            return 0.0

        # Compute average pairwise distance between the two physical partitions
        total_dist = 0.0
        count = 0
        for p1 in p_part1:
            for p2 in p_part2:
                total_dist += self.distance_matrix[p1][p2]
                count += 1
        avg_dist = total_dist / count if count > 0 else 1.0

        # Sum cross-partition interaction weights
        l_set1 = set(l_part1)
        l_set2 = set(l_part2)
        cross_weight = 0.0
        for (q1, q2), w in interaction_weight.items():
            if (q1 in l_set1 and q2 in l_set2) or (q1 in l_set2 and q2 in l_set1):
                cross_weight += w

        return cross_weight * avg_dist

    # --- Recursive bisection with assignment optimization ---
    lq_to_phys = {}
    used_physical = set()

    def recurse(lq_part, hw_part):
        n_lq = len(lq_part)
        if n_lq == 0:
            return

        # Base case: singleton
        if n_lq == 1:
            lq = lq_part[0]
            candidates = [p for p in hw_part if p not in used_physical]
            if not candidates:
                candidates = [p for p in physical_qubits if p not in used_physical]
            if candidates:
                best = max(candidates, key=lambda p: len(self.backend.get(p, [])))
                lq_to_phys[lq] = best
                used_physical.add(best)
            return

        # Step 1: Bisect logical partition via Fiedler vector of interaction Laplacian
        fl = fiedler_logical(lq_part)
        L1, L2 = bisect(lq_part, fl)

        # Step 2: Bisect hardware partition via Fiedler vector of coupling Laplacian
        fh = fiedler_hardware(hw_part)
        P1, P2 = bisect(hw_part, fh)

        # Balance: ensure each physical half can hold its logical half
        if len(P1) < len(L1) and len(P2) > len(L2):
            diff = len(L1) - len(P1)
            P1 = P1 + P2[:diff]
            P2 = P2[diff:]
        elif len(P2) < len(L2) and len(P1) > len(L1):
            diff = len(L2) - len(P2)
            P2 = P2 + P1[-diff:]
            P1 = P1[:-diff]

        # Step 3: Choose assignment minimizing cross-partition interaction * distance
        cost_direct = cross_partition_cost(L1, L2, P1, P2)
        cost_swapped = cross_partition_cost(L1, L2, P2, P1)

        if cost_swapped < cost_direct:
            recurse(L1, P2)
            recurse(L2, P1)
        else:
            recurse(L1, P1)
            recurse(L2, P2)

    # Step 4: Launch recursion
    recurse(logical_qubits, physical_qubits)

    # --- Build strict 1-to-1 bijection over all num_qubits indices ---
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