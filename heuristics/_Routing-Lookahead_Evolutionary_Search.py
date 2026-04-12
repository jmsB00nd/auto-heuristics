def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import random

    num_q = self.num_qubits

    # --- Identify logical qubits actually used in the circuit ---
    logical_qubits_used = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubits_used.add(q)
    logical_qubits_used = sorted(logical_qubits_used)
    n_logical = len(logical_qubits_used)

    # --- Identify valid physical qubits (those in the backend graph) ---
    physical_qubits = sorted(self.backend.keys())
    n_physical = len(physical_qubits)

    # --- Build interaction weight matrix for QAP seeds ---
    interaction_weights = {}
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            pair = (min(q1, q2), max(q1, q2))
            interaction_weights[pair] = interaction_weights.get(pair, 0) + 1

    # --- Collect first ~100 two-qubit gates in topological order for router fitness ---
    # Build a simple DAG from access/write_dict
    gate_ids = sorted(self.access.keys())
    two_q_gates = [g for g in gate_ids if len(self.access[g]) == 2]
    router_gates = two_q_gates[:100]

    # Build simple dependency info for router gates
    # Track last gate that touched each qubit for ordering
    def build_gate_order(all_gates, access):
        """Return 2q gates in dependency-respecting order with predecessors."""
        last_touch = {}  # qubit -> last gate id
        preds = {g: set() for g in all_gates if len(access[g]) == 2}
        ordered_2q = []
        for g in all_gates:
            qubits = access[g]
            dep_gates = set()
            for q in qubits:
                if q in last_touch:
                    dep_gates.add(last_touch[q])
            if len(qubits) == 2:
                # Only track 2q predecessors
                for dg in dep_gates:
                    if len(access[dg]) == 2:
                        preds[g].add(dg)
                ordered_2q.append(g)
            for q in qubits:
                last_touch[q] = g
        return ordered_2q, preds

    ordered_2q_gates, gate_preds = build_gate_order(gate_ids, self.access)
    # Limit to first 100
    router_gate_set = set(ordered_2q_gates[:100])

    # --- Lightweight SABRE-style router for fitness evaluation ---
    def evaluate_routing_cost(perm):
        """
        Given a permutation (logical->physical mapping as list of length num_q),
        simulate greedy routing on first 100 2q gates, return swap count.
        """
        # Build local mapping copies
        l2p = list(perm)
        p2l = [-1] * num_q
        for l, p in enumerate(l2p):
            if p >= 0:
                p2l[p] = l

        # Build local predecessor counts
        pred_count = {}
        succs = {}
        for g in router_gate_set:
            pred_count[g] = 0
            succs[g] = set()

        for g in router_gate_set:
            for pg in gate_preds.get(g, set()):
                if pg in router_gate_set:
                    pred_count[g] += 1
                    succs[pg].add(g)

        # Initialize front layer
        front = set()
        for g in router_gate_set:
            if pred_count[g] == 0:
                front.add(g)

        executed = set()
        swap_count = 0
        max_iters = len(router_gate_set) * 20  # safety bound
        iters = 0

        while front and iters < max_iters:
            iters += 1
            # Try to execute gates that are adjacent
            executed_this_round = []
            for g in list(front):
                q1, q2 = self.access[g]
                p1, p2 = l2p[q1], l2p[q2]
                if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                    executed_this_round.append(g)

            if executed_this_round:
                for g in executed_this_round:
                    front.discard(g)
                    executed.add(g)
                    for sg in succs.get(g, set()):
                        pred_count[sg] -= 1
                        if pred_count[sg] == 0:
                            front.add(sg)
                continue

            # No gate executable -> find best swap
            best_swap = None
            best_score = float('inf')

            # Collect active physical qubits from front layer
            active_phys = set()
            for g in front:
                q1, q2 = self.access[g]
                active_phys.add(l2p[q1])
                active_phys.add(l2p[q2])

            # Generate swap candidates
            for pq in active_phys:
                for neighbor in self.backend.get(pq, set()):
                    # Score this swap: sum of distances after swap for front layer gates
                    # Simulate swap
                    lq1 = p2l[pq]
                    lq2 = p2l[neighbor]
                    # After swap: lq1->neighbor, lq2->pq
                    score = 0.0
                    for g in front:
                        gq1, gq2 = self.access[g]
                        gp1 = l2p[gq1]
                        gp2 = l2p[gq2]
                        # Apply swap effect
                        if gp1 == pq:
                            gp1 = neighbor
                        elif gp1 == neighbor:
                            gp1 = pq
                        if gp2 == pq:
                            gp2 = neighbor
                        elif gp2 == neighbor:
                            gp2 = pq
                        score += self.distance_matrix[gp1][gp2]

                    if score < best_score:
                        best_score = score
                        best_swap = (pq, neighbor)

            if best_swap is None:
                break

            # Apply swap
            pq_a, pq_b = best_swap
            lq_a, lq_b = p2l[pq_a], p2l[pq_b]
            if lq_a >= 0:
                l2p[lq_a] = pq_b
            if lq_b >= 0:
                l2p[lq_b] = pq_a
            p2l[pq_a], p2l[pq_b] = lq_b, lq_a
            swap_count += 1

        return swap_count

    # --- QAP Frank-Wolfe seed generation ---
    def qap_frank_wolfe_seed(seed_val):
        """Generate a mapping via QAP relaxation + Hungarian projection."""
        rng = np.random.RandomState(seed_val)

        # Build flow matrix F (interaction weights between logical qubits)
        log_idx = {q: i for i, q in enumerate(logical_qubits_used)}
        F = np.zeros((n_logical, n_logical))
        for (q1, q2), w in interaction_weights.items():
            if q1 in log_idx and q2 in log_idx:
                i, j = log_idx[q1], log_idx[q2]
                F[i, j] = w
                F[j, i] = w

        # Build distance matrix D for physical qubits
        D = np.zeros((n_physical, n_physical))
        for i, p1 in enumerate(physical_qubits):
            for j, p2 in enumerate(physical_qubits):
                D[i, j] = self.distance_matrix[p1][p2]

        # Start with random doubly-stochastic matrix
        n = min(n_logical, n_physical)
        X = rng.random((n, n_physical))
        # Sinkhorn normalization (few iterations)
        for _ in range(20):
            X = X / (X.sum(axis=1, keepdims=True) + 1e-12)
            X = X / (X.sum(axis=0, keepdims=True) + 1e-12)

        # Frank-Wolfe iterations
        for fw_iter in range(15):
            # Gradient: dC/dX = F @ X @ D (QAP linearization)
            grad = F[:n, :n] @ X @ D
            # Solve linear assignment on gradient
            row_ind, col_ind = linear_sum_assignment(grad)
            # Build vertex Y
            Y = np.zeros_like(X)
            Y[row_ind, col_ind] = 1.0
            # Step size
            gamma = 2.0 / (fw_iter + 2)
            X = (1 - gamma) * X + gamma * Y

        # Final projection via Hungarian
        row_ind, col_ind = linear_sum_assignment(F[:n, :n] @ X @ D)

        # Build permutation
        perm = [-1] * num_q
        used_physical = set()
        for i, ci in zip(row_ind, col_ind):
            lq = logical_qubits_used[i]
            pq = physical_qubits[ci]
            perm[lq] = pq
            used_physical.add(pq)

        # Assign unmapped logical qubits to remaining physical qubits
        free_physical = [p for p in physical_qubits if p not in used_physical]
        unmapped_logical = [q for q in range(num_q) if perm[q] == -1]
        # Also add physical qubits not in backend but within range
        all_phys_available = set(range(num_q)) - used_physical
        free_physical = sorted(all_phys_available)

        for lq, pq in zip(unmapped_logical, free_physical):
            perm[lq] = pq

        return perm

    # --- Generate μ=5 seed population via QAP Frank-Wolfe ---
    MU = 5
    LAMBDA_PER_PARENT = 2
    GENERATIONS = 20
    NUM_TRANSPOSITIONS = 4  # 3-5 random adjacent transpositions

    population = []
    for seed_i in range(MU):
        perm = qap_frank_wolfe_seed(seed_i * 42 + 7)
        population.append(perm)

    # --- Mutation: apply 3-5 adjacent transpositions ---
    def mutate(perm, rng):
        child = list(perm)
        n_swaps = rng.randint(3, 6)  # 3 to 5 inclusive
        for _ in range(n_swaps):
            # Pick a random physical qubit and swap with a random neighbor
            pq = rng.choice(physical_qubits)
            neighbors = list(self.backend.get(pq, set()))
            if neighbors:
                neighbor = rng.choice(neighbors)
                # Swap in permutation: find which logical qubits map to pq and neighbor
                lq_a = -1
                lq_b = -1
                for lq in range(num_q):
                    if child[lq] == pq:
                        lq_a = lq
                    if child[lq] == neighbor:
                        lq_b = lq
                if lq_a >= 0 and lq_b >= 0:
                    child[lq_a], child[lq_b] = child[lq_b], child[lq_a]
        return child

    # --- Evolutionary loop ---
    rng = random.Random(12345)
    np_rng = np.random.RandomState(12345)

    # Evaluate initial population
    fitness = [(evaluate_routing_cost(p), i) for i, p in enumerate(population)]
    fitness.sort()

    for gen in range(GENERATIONS):
        offspring = []
        for parent_idx in range(min(MU, len(population))):
            parent = population[parent_idx]
            for _ in range(LAMBDA_PER_PARENT):
                child = mutate(parent, np_rng)
                offspring.append(child)

        # Evaluate offspring
        all_candidates = list(population) + offspring
        all_fitness = [(evaluate_routing_cost(c), i) for i, c in enumerate(all_candidates)]
        all_fitness.sort()

        # Select top MU
        new_population = []
        for _, idx in all_fitness[:MU]:
            new_population.append(list(all_candidates[idx]))
        population = new_population

    # --- Best individual ---
    best_perm = population[0]

    # --- Set mapping_dict and reverse_mapping_dict ---
    self.mapping_dict = list(best_perm)
    self.reverse_mapping_dict = [-1] * num_q
    for lq in range(num_q):
        pq = self.mapping_dict[lq]
        if 0 <= pq < num_q:
            self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        from src.utils.python_to_isl import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)