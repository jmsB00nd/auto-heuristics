def init_mapping(self):
    from collections import defaultdict, deque
    import numpy as np

    # --- Step 1: Collect logical qubits and build interaction affinity matrix ---
    logical_qubit_set = set()
    interaction_weight = defaultdict(lambda: defaultdict(float))

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            interaction_weight[q1][q2] += 1.0
            interaction_weight[q2][q1] += 1.0

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    n = len(logical_qubits)
    lq_index = {q: i for i, q in enumerate(logical_qubits)}

    # --- Step 2: Build similarity matrix for Affinity Propagation ---
    S = np.full((n, n), 0.0)
    for i, qi in enumerate(logical_qubits):
        for j, qj in enumerate(logical_qubits):
            if i != j:
                S[i][j] = interaction_weight[qi][qj]

    # Set preference (diagonal) to median of positive similarities
    off_diag = S[~np.eye(n, dtype=bool)]
    if len(off_diag) > 0 and np.any(off_diag > 0):
        preference = np.median(off_diag[off_diag > 0])
    else:
        preference = 0.0
    np.fill_diagonal(S, preference)

    # --- Step 3: Run Affinity Propagation message passing ---
    max_iter = 200
    damping = 0.7
    R = np.zeros((n, n))
    A = np.zeros((n, n))

    for iteration in range(max_iter):
        R_old = R.copy()
        A_old = A.copy()

        # Update responsibilities: R[i,k] = S[i,k] - max_{k'!=k}(A[i,k'] + S[i,k'])
        AS = A + S
        for i in range(n):
            row = AS[i].copy()
            for k in range(n):
                tmp = row[k]
                row[k] = -np.inf
                R[i, k] = S[i, k] - np.max(row)
                row[k] = tmp

        R = damping * R_old + (1 - damping) * R

        # Update availabilities
        for k in range(n):
            col = np.maximum(R[:, k], 0)
            col[k] = R[k, k]  # self-responsibility not clipped
            total = np.sum(col)
            for i in range(n):
                if i == k:
                    A[i, k] = total - col[k]
                else:
                    A[i, k] = min(0, total - col[i])

        A = damping * A_old + (1 - damping) * A

        if iteration > 10:
            if np.allclose(R, R_old, atol=1e-6) and np.allclose(A, A_old, atol=1e-6):
                break

    # --- Step 4: Extract exemplars and clusters ---
    exemplar_indices = np.where(np.diag(R + A) > 0)[0]

    if len(exemplar_indices) == 0:
        exemplar_indices = np.arange(n)

    clusters = defaultdict(list)
    exemplar_set = set(exemplar_indices.tolist())

    for i in range(n):
        if i in exemplar_set:
            clusters[i].append(i)
        else:
            best_ex = exemplar_indices[0]
            best_val = S[i, exemplar_indices[0]]
            for ex in exemplar_indices[1:]:
                if S[i, ex] > best_val:
                    best_val = S[i, ex]
                    best_ex = ex
            clusters[best_ex].append(i)

    # --- Step 5: Rank physical qubits by degree ---
    phys_degree = {p: len(self.backend[p]) for p in physical_qubits}
    phys_sorted_by_degree = sorted(physical_qubits, key=lambda p: -phys_degree[p])

    # --- Step 6: Map exemplars to high-degree physical qubits, members to neighbors ---
    sorted_exemplars = sorted(clusters.keys(), key=lambda ex: -len(clusters[ex]))

    assigned_physical = set()
    lq_to_phys = {}
    phys_idx = 0

    for ex_idx in sorted_exemplars:
        while phys_idx < len(phys_sorted_by_degree) and phys_sorted_by_degree[phys_idx] in assigned_physical:
            phys_idx += 1
        if phys_idx >= len(phys_sorted_by_degree):
            break

        ex_phys = phys_sorted_by_degree[phys_idx]
        ex_lq = logical_qubits[ex_idx]
        lq_to_phys[ex_lq] = ex_phys
        assigned_physical.add(ex_phys)

        members = [m for m in clusters[ex_idx] if m != ex_idx]
        members.sort(key=lambda m: -interaction_weight[logical_qubits[m]][ex_lq])

        # BFS from exemplar's physical qubit for nearby slots
        queue = deque()
        for neighbor in self.backend[ex_phys]:
            if neighbor not in assigned_physical:
                queue.append(neighbor)

        visited = {ex_phys}
        nearby_available = []
        while queue and len(nearby_available) < len(members):
            p = queue.popleft()
            if p in visited:
                continue
            visited.add(p)
            if p not in assigned_physical:
                nearby_available.append(p)
            for nb in self.backend[p]:
                if nb not in visited:
                    queue.append(nb)

        for i, m_idx in enumerate(members):
            m_lq = logical_qubits[m_idx]
            if i < len(nearby_available):
                lq_to_phys[m_lq] = nearby_available[i]
                assigned_physical.add(nearby_available[i])
            else:
                for p in physical_qubits:
                    if p not in assigned_physical:
                        lq_to_phys[m_lq] = p
                        assigned_physical.add(p)
                        break

    # --- Step 7: Build final mapping via bijection-preserving swaps ---
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