def init_mapping(self):
    """
    PASEM: Progressive Analytical SWAP Expectation Minimization

    Analytically computes E[SWAPs] = sum_{gates g in layers 0..K-1} w(layer) * dist(M[q1], M[q2])
    as a closed-form function of mapping M, then minimizes it via:
      1. Topological layer extraction (O(|gates|))
      2. Weighted interaction graph construction
      3. Greedy BFS embedding of the interaction graph onto the hardware topology
      4. Single-pass 2-opt local search refinement
    """
    from collections import defaultdict, deque

    num_q = self.num_qubits
    K = 10  # Topological depth horizon for SWAP expectation

    # ------------------------------------------------------------------ #
    # Step 1 – Extract 2-qubit gates in program order                     #
    # ------------------------------------------------------------------ #
    two_qubit_gates = sorted(
        [(g, qs[0], qs[1]) for g, qs in self.access.items() if len(qs) == 2],
        key=lambda x: x[0]
    )

    # ------------------------------------------------------------------ #
    # Step 2 – Assign gates to topological layers                         #
    # layer(g) = max(last_layer(q1), last_layer(q2)) + 1                 #
    # ------------------------------------------------------------------ #
    qubit_last_layer = {}
    gate_layer = {}
    for gate, q1, q2 in two_qubit_gates:
        l = max(qubit_last_layer.get(q1, -1), qubit_last_layer.get(q2, -1)) + 1
        gate_layer[gate] = l
        qubit_last_layer[q1] = l
        qubit_last_layer[q2] = l

    # ------------------------------------------------------------------ #
    # Step 3 – Build closed-form SWAP expectation weights                 #
    # E[SWAPs | M] ~ sum_{(q1,q2) in layers < K} w(layer) * dist(M[q1], M[q2]) #
    # w(layer) = 1 / (layer + 1)  (earlier layers contribute more)       #
    # ------------------------------------------------------------------ #
    interaction = defaultdict(float)
    for gate, q1, q2 in two_qubit_gates:
        if gate_layer[gate] < K:
            pair = (min(q1, q2), max(q1, q2))
            interaction[pair] += 1.0 / (gate_layer[gate] + 1.0)

    # Fallback to trivial mapping when no 2-qubit gates exist
    if not interaction:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Per-qubit adjacency list for O(degree) delta-cost computation
    qubit_adj = defaultdict(list)
    for (q1, q2), w in interaction.items():
        qubit_adj[q1].append((q2, w))
        qubit_adj[q2].append((q1, w))

    dm = self.distance_matrix
    dm_size = len(dm)

    def dist(p1, p2):
        if p1 < dm_size and p2 < dm_size:
            return dm[p1][p2]
        return float('inf')

    # ------------------------------------------------------------------ #
    # Step 4 – Greedy BFS embedding                                       #
    # Seed the most-interacting logical pair on the most central          #
    # adjacent physical pair, then expand by proximity on hardware.      #
    # ------------------------------------------------------------------ #
    ltp = [-1] * num_q   # logical  -> physical
    ptl = [-1] * num_q   # physical -> logical
    assigned_l = set()
    assigned_p = set()

    def assign(lq, pq):
        ltp[lq] = pq
        ptl[pq] = lq
        assigned_l.add(lq)
        assigned_p.add(pq)

    # Physical qubits valid for distance lookups
    physical_qubits = sorted(self.backend.keys())

    # Best seed physical edge: adjacent pair with maximum combined degree
    best_val = -1
    sp1, sp2 = physical_qubits[0], next(iter(self.backend[physical_qubits[0]]))
    for p in physical_qubits:
        for nb in self.backend[p]:
            val = len(self.backend[p]) + len(self.backend[nb])
            if val > best_val:
                best_val = val
                sp1, sp2 = p, nb

    seed_lq1, seed_lq2 = max(interaction, key=lambda p: interaction[p])
    assign(seed_lq1, sp1)
    assign(seed_lq2, sp2)

    # BFS expansion on the logical interaction graph
    bfs = deque([seed_lq1, seed_lq2])
    while bfs:
        lq = bfs.popleft()
        pq = ltp[lq]
        for (nlq, w) in sorted(qubit_adj[lq], key=lambda x: -x[1]):
            if nlq not in assigned_l:
                # Place this neighbor as close as possible on hardware
                candidates = [p for p in physical_qubits if p not in assigned_p]
                if not candidates:
                    candidates = [p for p in range(num_q) if p not in assigned_p]
                if candidates:
                    best_p = min(candidates, key=lambda p: dist(pq, p))
                    assign(nlq, best_p)
                    bfs.append(nlq)

    # Assign all remaining logical qubits (no interactions or orphaned)
    rem_l = [q for q in range(num_q) if q not in assigned_l]
    rem_p = [p for p in range(num_q) if p not in assigned_p]
    for lq, pq in zip(rem_l, rem_p):
        assign(lq, pq)

    # ------------------------------------------------------------------ #
    # Step 5 – Single-pass 2-opt local search                             #
    # For each pair (i, j), compute delta E[SWAPs] from swapping their   #
    # physical assignments. Accept if delta < 0. O(n^2 * max_degree).    #
    # ------------------------------------------------------------------ #
    for i in range(num_q):
        for j in range(i + 1, num_q):
            pi, pj = ltp[i], ltp[j]
            delta = 0.0
            # Contribution of i's interaction edges (excluding the i-j edge itself)
            for (nb, w) in qubit_adj[i]:
                if nb == j:
                    continue
                pn = ltp[nb]
                delta += w * (dist(pj, pn) - dist(pi, pn))
            # Contribution of j's interaction edges (excluding the i-j edge)
            for (nb, w) in qubit_adj[j]:
                if nb == i:
                    continue
                pn = ltp[nb]
                delta += w * (dist(pi, pn) - dist(pj, pn))
            if delta < -1e-9:
                ltp[i], ltp[j] = pj, pi
                ptl[pi] = j
                ptl[pj] = i

    self.mapping_dict = ltp
    self.reverse_mapping_dict = ptl

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)