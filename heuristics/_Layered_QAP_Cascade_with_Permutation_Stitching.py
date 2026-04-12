def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    from scipy.optimize import linear_sum_assignment

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)

    # -------------------------------------------------------------------
    # Step 1: Build DAG and topological ordering via Kahn's algorithm
    # -------------------------------------------------------------------
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

    all_gates = sorted(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_order = []
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in sorted(successors.get(g, set())):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # Identify 2-qubit gates in topological order
    topo_2q_gates = [g for g in topo_order if len(self.access[g]) == 2]

    # Collect logical qubits
    logical_qubits_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubits_set.add(q)
    logical_qubits = sorted(logical_qubits_set)

    # If no 2-qubit gates, use trivial mapping
    if not topo_2q_gates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # -------------------------------------------------------------------
    # Step 2: Partition 2-qubit gates into L=4 contiguous layers
    # -------------------------------------------------------------------
    L = 4
    n_2q = len(topo_2q_gates)
    layer_size = max(1, n_2q // L)
    layers = []
    for ell in range(L):
        start = ell * layer_size
        if ell == L - 1:
            end = n_2q  # last layer gets remainder
        else:
            end = start + layer_size
        if start < n_2q:
            layers.append(topo_2q_gates[start:end])

    # -------------------------------------------------------------------
    # Step 3: Build per-layer interaction matrices W_ell (n_phys x n_phys)
    # We use indices 0..n_phys-1 corresponding to logical qubits 0..n_phys-1
    # For logical qubits beyond n_phys, we still need them in the matrix.
    # Use num_q as the dimension (covers all logical and physical qubits).
    # -------------------------------------------------------------------
    N = num_q  # matrix dimension

    layer_W = []
    for layer_gates in layers:
        W = np.zeros((N, N), dtype=np.float64)
        for g in layer_gates:
            q1, q2 = self.access[g]
            W[q1, q2] += 1.0
            W[q2, q1] += 1.0
        layer_W.append(W)

    # Distance matrix as numpy array (N x N)
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i in self.distance_matrix and j in self.distance_matrix[i]:
                D[i, j] = self.distance_matrix[i][j]
            else:
                # For qubits not in backend, use large distance
                D[i, j] = N if i != j else 0

    # -------------------------------------------------------------------
    # Step 4: Frank-Wolfe QAP solver on Birkhoff polytope
    # -------------------------------------------------------------------
    def compute_qap_gradient(W, D, P):
        """Gradient of trace(W @ P @ D @ P^T) w.r.t. P = W^T @ P @ D^T + W @ P @ D"""
        return W.T @ P @ D.T + W @ P @ D

    def hungarian_projection(M):
        """Project matrix M onto permutation matrices via Hungarian algorithm.
        Find permutation P that minimizes trace(M^T @ P) = sum M_ij * P_ij.
        Since P is a permutation, this is a linear assignment: minimize sum M[i, sigma(i)].
        """
        row_ind, col_ind = linear_sum_assignment(M)
        P = np.zeros_like(M)
        P[row_ind, col_ind] = 1.0
        return P

    def solve_qap_frank_wolfe(W, D, P_prev=None, mu=0.0, max_iter=50):
        """
        Solve min trace(W P D P^T) + mu * ||P - P_prev||^2_F
        over the Birkhoff polytope using Frank-Wolfe with Hungarian projection.
        """
        # Initialize: if P_prev given, start from it; otherwise identity
        if P_prev is not None:
            P = P_prev.copy()
        else:
            P = np.eye(N, dtype=np.float64)

        for t in range(1, max_iter + 1):
            # Gradient of QAP objective
            grad = compute_qap_gradient(W, D, P)

            # Add penalty gradient if mu > 0
            if mu > 0 and P_prev is not None:
                grad += 2.0 * mu * (P - P_prev)

            # Linear minimization oracle: find permutation minimizing <grad, S>
            S = hungarian_projection(grad)

            # Step size (standard Frank-Wolfe diminishing step)
            gamma = 2.0 / (t + 2.0)

            # Update
            P = P + gamma * (S - P)

        # Final projection to nearest permutation
        P_final = hungarian_projection(-P)  # maximize <P, Perm> = minimize <-P, Perm>
        # Actually we want the permutation closest to P, which minimizes ||P - Perm||^2
        # = minimize -2<P, Perm> + const, so minimize <-P, Perm>
        # But hungarian minimizes cost, so we pass -P to get max <P, Perm>
        # Correction: linear_sum_assignment minimizes sum, so passing -P minimizes sum(-P[i,j]*Perm[i,j])
        # which maximizes sum(P[i,j]*Perm[i,j]) = <P, Perm>. This gives closest permutation.

        return P_final

    # -------------------------------------------------------------------
    # Step 5: Layered QAP Cascade
    # -------------------------------------------------------------------
    mu = 1.0  # continuity penalty weight

    # Layer 1: solve freely
    P_prev = solve_qap_frank_wolfe(layer_W[0], D, P_prev=None, mu=0.0, max_iter=60)
    P_first = P_prev.copy()  # This is pi_1, our final mapping

    # Layers 2..L: solve with continuity penalty
    for ell in range(1, len(layers)):
        P_prev = solve_qap_frank_wolfe(layer_W[ell], D, P_prev=P_prev, mu=mu, max_iter=40)

    # Note: The algorithm description says "the final mapping is pi_1"
    # but the cascade influences pi_1 only indirectly through the design.
    # Actually re-reading: "The final mapping is π₁ (the initial assignment)"
    # So we use P_first directly.

    # However, the cascade as described doesn't feed back to layer 1.
    # The novelty is that we solve layer 1 freely, then check if future layers
    # would prefer something different. A better interpretation: solve all layers
    # and use layer 1's solution since it's optimized for early gates.
    # 
    # Alternative: run cascade forward to get all P_ell, then run backward
    # cascade to refine P_1. But the description says "final mapping is π₁".
    # We'll use P_first as stated.

    # Extract permutation from P_first
    # P_first[i, j] = 1 means logical qubit i -> physical qubit j
    mapping = list(range(num_q))
    reverse_mapping = list(range(num_q))

    # Extract assignment from permutation matrix
    for i in range(N):
        for j in range(N):
            if P_first[i, j] > 0.5:
                mapping[i] = j
                reverse_mapping[j] = i
                break

    # -------------------------------------------------------------------
    # Step 6: Validate and fix any collisions (shouldn't happen with Hungarian)
    # -------------------------------------------------------------------
    used_phys = set()
    unmapped_logical = []
    for lq in range(num_q):
        pq = mapping[lq]
        if pq in used_phys:
            mapping[lq] = -1
            unmapped_logical.append(lq)
        else:
            used_phys.add(pq)

    free_phys = [pq for pq in range(num_q) if pq not in used_phys]
    for lq, pq in zip(unmapped_logical, free_phys):
        mapping[lq] = pq
        reverse_mapping[pq] = lq

    # Rebuild reverse mapping cleanly
    reverse_mapping = [0] * num_q
    for lq in range(num_q):
        reverse_mapping[mapping[lq]] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)