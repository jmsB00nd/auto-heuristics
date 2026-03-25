def init_mapping(self):
    """
    Fused Spectral-Seed + Routing-Local-Search (FSSRLS) initial placement.

    Phase 1 – SFQA (Spectral Fiedler Alignment):
      Compute the Fiedler vector (2nd eigenvector of the weighted Laplacian)
      for both the circuit interaction graph and the hardware graph.
      Sorting both by their Fiedler value and aligning them gives a
      globally structure-aware placement in O(n^3) (dominated by eigh).

    Phase 2 – Placement Local Search:
      Build a lightweight 2-qubit DAG to estimate the front layer and an
      extended lookahead layer (reusing the Qlosure distance/layer logic).
      Iteratively accept the best pairwise transposition of logical qubit
      assignments that reduces the Qlosure-style placement cost.
    """
    import numpy as np
    from collections import defaultdict

    # ------------------------------------------------------------------ #
    # Gather logical qubits and 2-qubit interaction weights               #
    # ------------------------------------------------------------------ #
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
    n_log = len(logical_qubits)
    n_phys = len(physical_qubits)

    # Trivial fallback: no logical qubits in circuit
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    lq_idx = {lq: i for i, lq in enumerate(logical_qubits)}
    pq_idx = {pq: i for i, pq in enumerate(physical_qubits)}

    # ================================================================== #
    # PHASE 1: Spectral Fiedler Alignment (SFQA)                         #
    # ================================================================== #

    def _fiedler_vector(nodes, neighbor_weights_fn):
        """
        Return the Fiedler vector (2nd eigenvector of the graph Laplacian)
        for a list of nodes.  Falls back to positional ordering on failure.
        neighbor_weights_fn(v) -> iterable of (neighbor, weight) pairs.
        """
        n = len(nodes)
        if n <= 2:
            return np.arange(n, dtype=float)
        local_idx = {v: i for i, v in enumerate(nodes)}
        L = np.zeros((n, n), dtype=float)
        for v in nodes:
            for u, w in neighbor_weights_fn(v):
                if u in local_idx:
                    i, j = local_idx[v], local_idx[u]
                    L[i, i] += w
                    L[i, j] -= w
        try:
            # eigh returns eigenvalues in ascending order for symmetric matrices
            _, vecs = np.linalg.eigh(L)
            return vecs[:, 1]          # Fiedler vector (2nd smallest eigenvalue)
        except Exception:
            return np.arange(n, dtype=float)

    # Logical interaction graph: weighted by gate co-occurrence count
    log_adj = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        if q1 in lq_idx and q2 in lq_idx:
            log_adj[q1][q2] = log_adj[q1].get(q2, 0.0) + w
            log_adj[q2][q1] = log_adj[q2].get(q1, 0.0) + w

    v_log = _fiedler_vector(logical_qubits, lambda lq: log_adj[lq].items())
    v_hw  = _fiedler_vector(physical_qubits,
                            lambda pq: ((nb, 1.0) for nb in self.backend[pq]
                                        if nb in pq_idx))

    # Align sorted Fiedler positions: sorted_log[i] -> sorted_phys[i]
    sorted_log  = [logical_qubits[i]  for i in np.argsort(v_log)]
    sorted_phys = [physical_qubits[i] for i in np.argsort(v_hw)]

    lq_to_phys = {}
    for i, lq in enumerate(sorted_log):
        if i < n_phys:
            lq_to_phys[lq] = sorted_phys[i]

    # Fill any remaining logical qubits (overflow guard)
    used_phys = set(lq_to_phys.values())
    remaining_phys = [p for p in physical_qubits if p not in used_phys]
    for lq in logical_qubits:
        if lq not in lq_to_phys and remaining_phys:
            lq_to_phys[lq] = remaining_phys.pop(0)

    # Build strict bijective mapping arrays via swap-in (identity base)
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq]            = target_phys
        mapping_dict[displaced_lq]  = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    # ================================================================== #
    # PHASE 2: Placement Local Search (Qlosure-style cost in placement    #
    # space, iterative best-improvement transposition)                    #
    # ================================================================== #

    # Collect 2-qubit gates in program order
    gates_2q_ordered = sorted(
        [(gate, qubits[0], qubits[1])
         for gate, qubits in self.access.items() if len(qubits) == 2]
    )

    if not gates_2q_ordered:
        self.mapping_dict         = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    gate_ids_2q    = [g for g, _, _ in gates_2q_ordered]
    gate_qubits_2q = {g: (q1, q2) for g, q1, q2 in gates_2q_ordered}

    # Build lightweight 2q successor DAG:
    # Each gate on qubit q succeeds the last 2q gate that touched q.
    successors_2q  = defaultdict(set)
    pred_count_2q  = {g: 0 for g in gate_ids_2q}
    last_2q_on_q   = {}                      # qubit -> last 2q gate id

    for gate, q1, q2 in gates_2q_ordered:
        for q in (q1, q2):
            if q in last_2q_on_q:
                prev = last_2q_on_q[q]
                if gate not in successors_2q[prev]:
                    successors_2q[prev].add(gate)
                    pred_count_2q[gate] += 1
        last_2q_on_q[q1] = gate
        last_2q_on_q[q2] = gate

    # Front layer: 2q gates with no 2q predecessors
    front_2q = [g for g in gate_ids_2q if pred_count_2q[g] == 0]
    n_front  = len(front_2q) or 1

    # Extended layer: BFS from front (Qlosure look-ahead, depth-discounted)
    LOOKAHEAD   = min(30, len(gate_ids_2q))
    extended_2q = []                         # list of (gate_id, bfs_depth)
    visited_ext = set(front_2q)
    bfs_queue   = [(g, 1) for g in front_2q]
    head        = 0

    while head < len(bfs_queue) and len(extended_2q) < LOOKAHEAD:
        g, depth = bfs_queue[head]; head += 1
        for succ in successors_2q[g]:
            if succ not in visited_ext:
                visited_ext.add(succ)
                extended_2q.append((succ, depth + 1))
                bfs_queue.append((succ, depth + 1))

    n_ext = len(extended_2q) or 1

    def _placement_cost(md):
        """
        Qlosure-style placement cost:
          H = (sum_{g in front} dist(md[q1], md[q2])) / |front|
            + (sum_{g in ext}   dist(md[q1], md[q2]) / depth_g) / |ext|
        """
        f = sum(
            self.distance_matrix[md[gate_qubits_2q[g][0]]][md[gate_qubits_2q[g][1]]]
            for g in front_2q
        )
        e = sum(
            self.distance_matrix[md[gate_qubits_2q[g][0]]][md[gate_qubits_2q[g][1]]] / d
            for g, d in extended_2q
        )
        return f / n_front + e / n_ext

    # Iterative best-improvement local search over logical-qubit transpositions.
    # Budget: O(n_log) outer iterations x O(n_log^2) pair evaluations.
    MAX_LS_ITER = max(2, n_log)

    for _ in range(MAX_LS_ITER):
        current_cost = _placement_cost(mapping_dict)
        best_gain    = 1e-9           # strict improvement threshold
        best_swap    = None

        for i in range(n_log):
            lq1 = logical_qubits[i]
            p1  = mapping_dict[lq1]
            for j in range(i + 1, n_log):
                lq2 = logical_qubits[j]
                p2  = mapping_dict[lq2]

                # Tentative transposition (no copy needed — we revert immediately)
                mapping_dict[lq1] = p2
                mapping_dict[lq2] = p1

                gain = current_cost - _placement_cost(mapping_dict)

                # Revert
                mapping_dict[lq1] = p1
                mapping_dict[lq2] = p2

                if gain > best_gain:
                    best_gain = gain
                    best_swap = (lq1, lq2)

        if best_swap is None:
            break                      # converged to local optimum

        lq1, lq2 = best_swap
        p1, p2   = mapping_dict[lq1], mapping_dict[lq2]
        mapping_dict[lq1]          = p2
        mapping_dict[lq2]          = p1
        reverse_mapping_dict[p1]   = lq2
        reverse_mapping_dict[p2]   = lq1

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)