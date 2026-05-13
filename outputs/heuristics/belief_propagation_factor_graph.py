def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = int(self.num_qubits)
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # ---- 1. Collect 2-qubit interactions (QIG edges) from self.access ----
    qig_w = defaultdict(float)
    active = set()
    for _, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            qig_w[key] += 1.0
            active.add(a)
            active.add(b)

    # No interactions -> identity
    if not qig_w:
        for i in range(N):
            self.mapping_dict[i] = i
            self.reverse_mapping_dict[i] = i
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    logicals = sorted(active)
    # Drop any out-of-range ids defensively
    logicals = [l for l in logicals if 0 <= l < N]
    L = len(logicals)
    l2i = {l: i for i, l in enumerate(logicals)}

    edges = []
    neighbors = defaultdict(list)
    for (a, b), w in qig_w.items():
        if a in l2i and b in l2i:
            i, j = l2i[a], l2i[b]
            edges.append((i, j, float(w)))
            neighbors[i].append((j, float(w)))
            neighbors[j].append((i, float(w)))

    # ---- Distance matrix ----
    try:
        D = np.asarray(self.distance_matrix, dtype=float)
        if D.shape[0] != N or D.shape[1] != N:
            raise ValueError
    except Exception:
        D = None

    if D is None or L == 0 or not edges:
        # Fall back to identity bijection
        used = set()
        for i in range(N):
            self.mapping_dict[i] = i
            self.reverse_mapping_dict[i] = i
            used.add(i)
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # ---- 2. Unary potentials ----
    cent = np.zeros(N, dtype=float)
    pc = getattr(self, 'physical_centrality', None)
    if isinstance(pc, dict):
        for p in range(N):
            cent[p] = float(pc.get(p, 0.0))
    if cent.max() > 0:
        cent_n = cent / cent.max()
    else:
        cent_n = np.zeros(N)

    act = np.zeros(L, dtype=float)
    la = getattr(self, 'logical_activity', None)
    for i, l in enumerate(logicals):
        if la is not None:
            try:
                act[i] = float(la[l])
            except Exception:
                act[i] = 0.0
    if act.max() > 0:
        act_n = act / act.max()
    else:
        act_n = np.ones(L)

    unary = np.outer(act_n, cent_n)  # shape (L, N)

    # ---- 3-4. Loopy max-sum belief propagation ----
    INF_PEN = 1e9
    diag_pen = INF_PEN * np.eye(N)

    msgs = {}
    for i, j, _ in edges:
        msgs[(i, j)] = np.zeros(N, dtype=float)
        msgs[(j, i)] = np.zeros(N, dtype=float)

    T = 8
    damping = 0.5

    def _outgoing(src, dst, w, msgs_in):
        incoming = unary[src].copy()
        for k, _w in neighbors[src]:
            if k != dst:
                incoming = incoming + msgs_in[(k, src)]
        # m_{src->dst}(p_dst) = max_{p_src} [ incoming[p_src] - w*D[p_src,p_dst] - INF*(p_src==p_dst) ]
        score = incoming[:, None] - w * D - diag_pen
        m = score.max(axis=0)
        m = m - m.max()  # normalize for stability
        return m

    for _ in range(T):
        new_msgs = {}
        for i, j, w in edges:
            m_ij = _outgoing(i, j, w, msgs)
            m_ji = _outgoing(j, i, w, msgs)
            new_msgs[(i, j)] = damping * msgs[(i, j)] + (1.0 - damping) * m_ij
            new_msgs[(j, i)] = damping * msgs[(j, i)] + (1.0 - damping) * m_ji
        msgs = new_msgs

    # ---- 5. Final beliefs and Hungarian decode ----
    beliefs = unary.copy()
    for i in range(L):
        for k, _w in neighbors[i]:
            beliefs[i] += msgs[(k, i)]

    assignment = [-1] * L
    used_phys = set()
    try:
        from scipy.optimize import linear_sum_assignment
        cost = -beliefs  # minimize cost = maximize belief
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            assignment[int(r)] = int(c)
            used_phys.add(int(c))
    except Exception:
        # Greedy conflict resolution: place logicals in decreasing belief-peak order
        order = sorted(range(L), key=lambda i: -float(beliefs[i].max()))
        for i in order:
            for p in np.argsort(-beliefs[i]):
                p = int(p)
                if p not in used_phys:
                    assignment[i] = p
                    used_phys.add(p)
                    break

    mapping = [-1] * N
    rmapping = [-1] * N
    for idx, l in enumerate(logicals):
        p = assignment[idx]
        if 0 <= p < N and rmapping[p] == -1 and mapping[l] == -1:
            mapping[l] = p
            rmapping[p] = l

    # ---- 6. Fill remaining logicals onto unused physicals (identity preferred) ----
    unused_phys = set(p for p in range(N) if rmapping[p] == -1)
    for l in range(N):
        if mapping[l] != -1:
            continue
        if l in unused_phys:
            mapping[l] = l
            rmapping[l] = l
            unused_phys.discard(l)
    for l in range(N):
        if mapping[l] != -1:
            continue
        p = next(iter(unused_phys))
        mapping[l] = p
        rmapping[p] = l
        unused_phys.discard(p)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = rmapping

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)