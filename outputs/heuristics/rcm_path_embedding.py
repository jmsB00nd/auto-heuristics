def init_mapping(self):
    import numpy as np

    N = self.num_qubits

    # --- 1. Weighted interaction matrix over logical qubits (from self.access) ---
    W = np.zeros((N, N), dtype=float)
    logical_used = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if 0 <= a < N and 0 <= b < N and a != b:
                W[a, b] += 1.0
                W[b, a] += 1.0
                logical_used.add(a)
                logical_used.add(b)

    # --- 2. RCM ordering on W ---
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        # Add tiny self-loop diagonal so isolated nodes are still included by RCM
        W_for_rcm = W.copy()
        np.fill_diagonal(W_for_rcm, np.maximum(np.diag(W_for_rcm), 1e-12))
        rcm_perm = list(reverse_cuthill_mckee(csr_matrix(W_for_rcm), symmetric_mode=True))
    except Exception:
        # Fallback: degree-descending then identity for the rest
        deg = W.sum(axis=1)
        rcm_perm = list(np.argsort(-deg))

    # Ensure rcm_perm is a full permutation of [0, N)
    seen = set(rcm_perm)
    if len(seen) != N:
        for q in range(N):
            if q not in seen:
                rcm_perm.append(q)
                seen.add(q)
        rcm_perm = rcm_perm[:N]

    # Prioritize circuit-active logical qubits at the front (preserve RCM order within)
    active_first = [q for q in rcm_perm if q in logical_used]
    inactive_tail = [q for q in rcm_perm if q not in logical_used]
    logical_order = active_first + inactive_tail

    # --- 3. Long greedy walk on the coupling graph ---
    def neighbors_of(p):
        try:
            return list(self.backend[p])
        except Exception:
            return []

    # Peripheral start: node with maximum eccentricity in self.distance_matrix
    start = 0
    best_ecc = -1
    for p in range(N):
        row = self.distance_matrix[p]
        try:
            ecc = max(row)
        except Exception:
            ecc = 0
        if ecc > best_ecc:
            best_ecc = ecc
            start = p

    visited = {start}
    walk = [start]
    current = start
    while True:
        cands = [n for n in neighbors_of(current) if n not in visited]
        if cands:
            pick = max(cands, key=lambda x: self.distance_matrix[start][x])
            walk.append(pick)
            visited.add(pick)
            current = pick
            continue
        # backtrack along walk to find any extension
        extended = False
        for node in reversed(walk):
            cands2 = [n for n in neighbors_of(node) if n not in visited]
            if cands2:
                pick = max(cands2, key=lambda x: self.distance_matrix[start][x])
                walk.append(pick)
                visited.add(pick)
                current = pick
                extended = True
                break
        if not extended:
            break

    # Pad walk with any physical qubits that the coupling graph never reached
    for p in range(N):
        if p not in visited:
            walk.append(p)
            visited.add(p)
    walk = walk[:N]

    # --- 4. Drape RCM-ordered logical qubits onto walk ---
    self.mapping_dict = [0] * N
    self.reverse_mapping_dict = [0] * N
    used_phys = set()
    assigned_logical = set()
    for i, logical in enumerate(logical_order):
        if i >= N:
            break
        phys = walk[i]
        self.mapping_dict[logical] = phys
        self.reverse_mapping_dict[phys] = logical
        used_phys.add(phys)
        assigned_logical.add(logical)

    # --- 5. Identity-style fallback for any leftover logical qubits ---
    remaining_logical = [q for q in range(N) if q not in assigned_logical]
    remaining_phys = [p for p in range(N) if p not in used_phys]
    for l, p in zip(remaining_logical, remaining_phys):
        self.mapping_dict[l] = p
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)