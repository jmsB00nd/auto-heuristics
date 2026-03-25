def init_mapping(self):
    """
    PageRank Dual Rank Matching (PRDM):

    Compute PageRank on both the directed logical interaction graph and the
    undirected hardware graph.  High-PageRank logical qubits (circuit flow
    hubs) are matched to high-PageRank physical qubits (connectivity hubs),
    descending rank order.  This exploits the structural duality between
    circuit centrality and hardware centrality to minimise routing distance.
    """
    from collections import defaultdict

    # ------------------------------------------------------------------ #
    # Step 1 – Collect logical qubits                                     #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
    for qubits in self.access.values():
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits  = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback: trivial identity if no gates are present
    if not logical_qubits:
        self.mapping_dict         = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 2 – Build directed, weighted logical interaction graph         #
    #                                                                     #
    # For each 2-qubit gate (ordered by gate key = temporal position),    #
    # add a directed edge  q1 -> q2  with weight  w = 1/(rank+1).        #
    # Earlier gates carry higher weight: they are the first constraints   #
    # the router must satisfy, so their qubit pairs are most critical.    #
    # ------------------------------------------------------------------ #
    sorted_gates = sorted(self.access.keys())   # temporal order by key

    # logical_out[src][dst] = total directed weight  (src -> dst)
    logical_out = defaultdict(lambda: defaultdict(float))

    for rank, gate in enumerate(sorted_gates):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            w = 1.0 / (rank + 1)               # recency bias: earlier = heavier
            logical_out[q1][q2] += w

    # Reverse index for efficient in-edge summation during PageRank
    logical_in = defaultdict(lambda: defaultdict(float))
    for src, targets in logical_out.items():
        for dst, w in targets.items():
            logical_in[dst][src] += w

    # ------------------------------------------------------------------ #
    # Step 3 – PageRank on the directed logical graph                     #
    #                                                                     #
    # PR(q) = (1-d)/N + d * Σ_{u→q}  PR(u) * w(u,q) / Σ_v w(u,v)       #
    # ------------------------------------------------------------------ #
    d        = 0.85
    max_iter = 100
    tol      = 1e-8
    n_l      = len(logical_qubits)

    logical_pr = {q: 1.0 / n_l for q in logical_qubits}

    for _ in range(max_iter):
        new_pr = {}
        for q in logical_qubits:
            rank_sum = 0.0
            for src, w_in in logical_in[q].items():
                out_total = sum(logical_out[src].values())
                if out_total > 0.0:
                    rank_sum += logical_pr[src] * (w_in / out_total)
            new_pr[q] = (1.0 - d) / n_l + d * rank_sum

        diff = sum(abs(new_pr[q] - logical_pr[q]) for q in logical_qubits)
        logical_pr = new_pr
        if diff < tol:
            break

    # ------------------------------------------------------------------ #
    # Step 4 – PageRank on the hardware graph                             #
    #                                                                     #
    # Undirected: each edge is bidirectional with uniform weight.         #
    # High-PageRank physical qubit = high connectivity = routing hub.     #
    # ------------------------------------------------------------------ #
    n_p = len(physical_qubits)
    physical_pr = {p: 1.0 / n_p for p in physical_qubits}

    for _ in range(max_iter):
        new_pr = {}
        for p in physical_qubits:
            neighbors = self.backend[p]
            rank_sum  = 0.0
            for nb in neighbors:
                out_deg = len(self.backend[nb])
                if out_deg > 0:
                    rank_sum += physical_pr[nb] / out_deg
            new_pr[p] = (1.0 - d) / n_p + d * rank_sum

        diff = sum(abs(new_pr[p] - physical_pr[p]) for p in physical_qubits)
        physical_pr = new_pr
        if diff < tol:
            break

    # ------------------------------------------------------------------ #
    # Step 5 – Greedy rank-descending matching                            #
    #                                                                     #
    # Sort both lists by PageRank (highest first) and zip.               #
    # Circuit hubs land on hardware hubs → maximum routing flexibility   #
    # for the gates that matter most.                                     #
    # ------------------------------------------------------------------ #
    sorted_logical  = sorted(logical_qubits,  key=lambda q: logical_pr[q],  reverse=True)
    sorted_physical = sorted(physical_qubits, key=lambda p: physical_pr[p], reverse=True)

    lq_to_phys = {lq: sorted_physical[i] for i, lq in enumerate(sorted_logical)}

    # ------------------------------------------------------------------ #
    # Step 6 – Build strict 1-to-1 bijection over all num_qubits indices #
    #                                                                     #
    # Start from identity, apply each assignment via an in-place swap,   #
    # displacing the qubit currently occupying the target slot.           #
    # Guarantees a valid permutation even for non-contiguous qubit IDs.  #
    # ------------------------------------------------------------------ #
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq]                   = target_phys
        mapping_dict[displaced_lq]         = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)