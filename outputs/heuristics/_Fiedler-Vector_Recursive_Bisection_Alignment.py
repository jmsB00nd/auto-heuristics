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
            interaction_weight[key] += 1

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback to trivial identity mapping if circuit has no gates
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Helper: Fiedler vector for a logical subgraph (weighted circuit Laplacian) ---
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
            _, vecs = np.linalg.eigh(L)
            fvec = vecs[:, 1]
        except Exception:
            # Fallback: split by index order
            fvec = np.arange(n, dtype=float)
        return {nodes[i]: fvec[i] for i in range(n)}

    # --- Helper: Fiedler vector for a hardware subgraph (unit-weight topology Laplacian) ---
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
            _, vecs = np.linalg.eigh(L)
            fvec = vecs[:, 1]
        except Exception:
            fvec = np.arange(n, dtype=float)
        return {hw_nodes[i]: fvec[i] for i in range(n)}

    # --- Helper: split nodes into two halves by median of Fiedler values ---
    def bisect(nodes, fiedler_dict):
        if len(nodes) <= 1:
            return nodes[:], []
        vals = np.array([fiedler_dict[n] for n in nodes])
        median_val = float(np.median(vals))
        left = [n for n in nodes if fiedler_dict[n] <= median_val]
        right = [n for n in nodes if fiedler_dict[n] > median_val]
        # Degenerate guard: ensure both sides non-empty
        if not right:
            mid = len(nodes) // 2
            left, right = nodes[:mid], nodes[mid:]
        elif not left:
            mid = len(nodes) // 2
            left, right = nodes[:mid], nodes[mid:]
        return left, right

    # --- Recursive Fiedler bisection alignment ---
    lq_to_phys = {}
    used_physical = set()

    def recurse(lq_part, hw_part):
        n_lq = len(lq_part)
        if n_lq == 0:
            return

        if n_lq == 1:
            # Base case: assign to highest-degree available physical qubit in hw_part
            lq = lq_part[0]
            candidates = [p for p in hw_part if p not in used_physical]
            if not candidates:
                candidates = [p for p in physical_qubits if p not in used_physical]
            if candidates:
                best = max(candidates, key=lambda p: len(self.backend.get(p, [])))
                lq_to_phys[lq] = best
                used_physical.add(best)
            return

        # Bisect logical partition using circuit Fiedler vector
        fl = fiedler_logical(lq_part)
        left_lq, right_lq = bisect(lq_part, fl)

        # Bisect hardware partition using topology Fiedler vector
        fh = fiedler_hardware(hw_part)
        left_hw, right_hw = bisect(hw_part, fh)

        # Align partition trees and recurse
        recurse(left_lq, left_hw)
        recurse(right_lq, right_hw)

    recurse(logical_qubits, physical_qubits)

    # --- Build strict 1-to-1 bijection over all num_qubits indices ---
    # Start from identity, apply spectral assignments via in-place swaps
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