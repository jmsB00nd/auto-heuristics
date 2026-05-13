def init_mapping(self):
    import math
    import random

    N = self.num_qubits
    D = self.distance_matrix

    # Build logical interaction weights from self.access (DAG not yet built).
    W = [dict() for _ in range(N)]
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig is not None:
        for i in range(N):
            row = qig.get(i, {}) if hasattr(qig, "get") else qig[i]
            for j, w in row.items():
                if 0 <= j < N and j != i and w:
                    W[i][j] = float(w)
    else:
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if 0 <= a < N and 0 <= b < N and a != b:
                    W[a][b] = W[a].get(b, 0.0) + 1.0
                    W[b][a] = W[b].get(a, 0.0) + 1.0

    # Neighbor lists for cheap incremental cost evaluation.
    neighbors = [list(W[i].items()) for i in range(N)]

    def total_cost(pi):
        c = 0.0
        for i in range(N):
            pi_i = pi[i]
            for j, w in neighbors[i]:
                if j > i:
                    c += w * D[pi_i][pi[j]]
        return c

    def partial_cost(pi, i):
        pi_i = pi[i]
        c = 0.0
        for j, w in neighbors[i]:
            c += w * D[pi_i][pi[j]]
        return c

    pi = list(range(N))
    random.shuffle(pi)
    inv_pi = [0] * N
    for k in range(N):
        inv_pi[pi[k]] = k

    cur_cost = total_cost(pi)
    best_pi = list(pi)
    best_cost = cur_cost

    # Estimate initial temperature from typical positive ΔC magnitudes.
    sample_deltas = []
    for _ in range(min(50, max(5, N))):
        i = random.randrange(N)
        j = random.randrange(N)
        if i == j:
            continue
        before = partial_cost(pi, i) + partial_cost(pi, j)
        # subtract double-counted W[i][j] term once (it's symmetric in pi swap)
        wij = W[i].get(j, 0.0)
        before -= wij * D[pi[i]][pi[j]]
        pi[i], pi[j] = pi[j], pi[i]
        after = partial_cost(pi, i) + partial_cost(pi, j)
        after -= wij * D[pi[i]][pi[j]]
        pi[i], pi[j] = pi[j], pi[i]
        d = abs(after - before)
        if d > 0:
            sample_deltas.append(d)
    T0 = (sum(sample_deltas) / len(sample_deltas)) if sample_deltas else 1.0
    T0 = max(T0, 1e-6)
    T_min = T0 * 1e-4

    iters = max(200, 50 * N * max(1, int(math.log2(max(2, N)))))
    iters = min(iters, 20000)
    alpha = (T_min / T0) ** (1.0 / max(1, iters))
    T = T0

    for _ in range(iters):
        i = random.randrange(N)
        j = random.randrange(N)
        if i == j:
            T *= alpha
            continue
        wij = W[i].get(j, 0.0)
        before = partial_cost(pi, i) + partial_cost(pi, j) - wij * D[pi[i]][pi[j]]
        pi[i], pi[j] = pi[j], pi[i]
        after = partial_cost(pi, i) + partial_cost(pi, j) - wij * D[pi[i]][pi[j]]
        delta = after - before
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            cur_cost += delta
            inv_pi[pi[i]] = i
            inv_pi[pi[j]] = j
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_pi = list(pi)
        else:
            pi[i], pi[j] = pi[j], pi[i]
        T *= alpha

    self.mapping_dict = list(best_pi)
    self.reverse_mapping_dict = [0] * N
    seen = set()
    for L in range(N):
        p = self.mapping_dict[L]
        if 0 <= p < N and p not in seen:
            self.reverse_mapping_dict[p] = L
            seen.add(p)
        else:
            self.mapping_dict[L] = -1
    unused_phys = [p for p in range(N) if p not in seen]
    for L in range(N):
        if self.mapping_dict[L] == -1:
            p = unused_phys.pop()
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)