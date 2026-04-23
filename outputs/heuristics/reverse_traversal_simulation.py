def init_mapping(self):
    from collections import deque

    N = self.num_qubits

    mapping = list(range(N))
    rev = list(range(N))

    def neighbors_of(x):
        if hasattr(self.backend, 'get'):
            return self.backend.get(x, [])
        try:
            return self.backend[x]
        except Exception:
            return []

    def shortest_path(u, v):
        if u == v:
            return [u]
        prev = {u: None}
        dq = deque([u])
        found = False
        while dq:
            x = dq.popleft()
            if x == v:
                found = True
                break
            for y in neighbors_of(x):
                if y not in prev:
                    prev[y] = x
                    dq.append(y)
        if not found and v not in prev:
            return None
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    try:
        sorted_gates = sorted(self.access.keys())
    except TypeError:
        sorted_gates = list(self.access.keys())

    forward_swaps = []

    for gate_id in sorted_gates:
        qubits = self.access[gate_id]
        if not qubits or len(qubits) != 2:
            continue
        q1, q2 = qubits[0], qubits[1]
        if q1 < 0 or q2 < 0 or q1 >= N or q2 >= N or q1 == q2:
            continue
        p1, p2 = mapping[q1], mapping[q2]
        if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
            continue
        path = shortest_path(p1, p2)
        if path is None or len(path) < 3:
            continue
        for i in range(len(path) - 2):
            a, b = path[i], path[i + 1]
            la, lb = rev[a], rev[b]
            mapping[la], mapping[lb] = b, a
            rev[a], rev[b] = lb, la
            forward_swaps.append((a, b))

    for a, b in reversed(forward_swaps):
        if a == b:
            continue

    self.mapping_dict = list(mapping)
    self.reverse_mapping_dict = list(rev)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)