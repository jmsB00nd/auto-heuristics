# Idea: \_NAME: Graph Bottleneck Avoidance Cost (GBAC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on multiplier_n45__462CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    from collections import deque

    W = 1.0
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Min Vertex Cut via Node-Splitting + Dinic's Max-Flow ---
    # Cache is keyed on (s, t) canonical pair; topology never changes.
    if not hasattr(self, '_gbac_cut_cache'):
        self._gbac_cut_cache = {}

    def compute_min_vertex_cut(s, t):
        """
        Min vertex cut between physical qubits s and t.
        Splits every interior node v into (v_in=2v, v_out=2v+1) with
        unit capacity on the internal edge, giving a flow-equivalent
        of vertex connectivity by Menger's theorem.
        """
        key = (min(s, t), max(s, t))
        if key in self._gbac_cut_cache:
            return self._gbac_cut_cache[key]

        if s == t:
            self._gbac_cut_cache[key] = self.num_qubits  # effectively infinite
            return self._gbac_cut_cache[key]

        n   = self.num_qubits
        INF = n + 1            # upper bound on any vertex cut

        # Build Dinic adjacency list.
        # Edge format: [to, capacity, reverse_index]
        num_nodes = 2 * n
        adj = [[] for _ in range(num_nodes)]

        def add_edge(u, v, cap):
            adj[u].append([v, cap, len(adj[v])])
            adj[v].append([u, 0,   len(adj[u]) - 1])

        # Internal node edges (vertex capacity = 1 for interior nodes)
        for v in range(n):
            cap = INF if (v == s or v == t) else 1
            add_edge(2 * v, 2 * v + 1, cap)

        # Hardware graph edges (arc capacity = INF, both directions)
        for u in range(n):
            for v in self.backend[u]:
                if u < v:
                    add_edge(2 * u + 1, 2 * v,     INF)
                    add_edge(2 * v + 1, 2 * u,     INF)

        source = 2 * s + 1   # s_out  (already past s's own capacity)
        sink   = 2 * t       # t_in   (block at t's internal edge)

        # --- Dinic's BFS: build level graph ---
        def bfs():
            level = [-1] * num_nodes
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v, cap, _ in adj[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            return level

        # --- Dinic's DFS: send blocking flow ---
        def dfs(u, pushed, level, it):
            if u == sink:
                return pushed
            while it[u] < len(adj[u]):
                e = adj[u][it[u]]
                v, cap, rev = e
                if cap > 0 and level[v] == level[u] + 1:
                    d = dfs(v, min(pushed, cap), level, it)
                    if d > 0:
                        e[1]          -= d
                        adj[v][rev][1] += d
                        return d
                it[u] += 1
            return 0

        max_flow = 0
        while True:
            level = bfs()
            if level[sink] < 0:
                break
            it = [0] * num_nodes
            while True:
                f = dfs(source, INF, level, it)
                if f == 0:
                    break
                max_flow += f

        result = max(max_flow, 1)          # guard against isolated nodes
        self._gbac_cut_cache[key] = result
        return result

    # --- Cost accumulation ---
    def bottleneck_weight(Q1, Q2):
        """w(g) = 1 / min_cut(Q1, Q2)  — higher cost through chokepoints."""
        if Q1 == Q2:
            return 0.0
        return 1.0 / compute_min_vertex_cut(Q1, Q2)

    # Front-layer contribution
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        w    = bottleneck_weight(Q1, Q2)
        dist = self.distance_matrix[Q1][Q2]
        f_cost += w * dist

    # Extended-layer (lookahead) contribution
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        layer_idx = self.extended_layer_index.get(g, 0) + 1
        w    = bottleneck_weight(Q1, Q2)
        dist = self.distance_matrix[Q1][Q2]
        e_cost += w * dist / layer_idx

    # H_GBAC = max_decay × (Σ_F / |F|  +  W × Σ_E / |E|)
    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H