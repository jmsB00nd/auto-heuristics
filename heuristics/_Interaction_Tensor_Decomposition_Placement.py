def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict, deque

    num_qubits = self.num_qubits

    # Step 0: Identify logical qubits and 2-qubit gates
    logical_qubits_used = set()
    two_qubit_gates = {}
    for gate_id, qubits in self.access.items():
        logical_qubits_used.update(qubits)
        if len(qubits) == 2:
            two_qubit_gates[gate_id] = (qubits[0], qubits[1])

    if not two_qubit_gates:
        self.mapping_dict = list(range(num_qubits))
        self.reverse_mapping_dict = list(range(num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Step 1: Build DAG layers via topological BFS
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    last_writer = {}

    gate_ids_sorted = sorted(self.access.keys())
    for gate_id in gate_ids_sorted:
        qubits = self.access[gate_id]
        write_qubits = self.write_dict.get(gate_id, [])
        for q in qubits:
            if q in last_writer:
                pred = last_writer[q]
                if pred != gate_id:
                    successors[pred].add(gate_id)
                    predecessors[gate_id].add(pred)
        for q in write_qubits:
            last_writer[q] = gate_id

    in_degree = {g: len(predecessors[g]) for g in gate_ids_sorted}
    queue = deque()
    gate_layer = {}
    for g in gate_ids_sorted:
        if in_degree.get(g, 0) == 0:
            queue.append(g)
            gate_layer[g] = 0

    while queue:
        g = queue.popleft()
        for s in successors[g]:
            in_degree[s] -= 1
            gate_layer[s] = max(gate_layer.get(s, 0), gate_layer[g] + 1)
            if in_degree[s] == 0:
                queue.append(s)

    # Step 2: Build interaction tensor T[i][j][l]
    lq_list = sorted(logical_qubits_used)
    lq_index = {q: idx for idx, q in enumerate(lq_list)}
    n_logical = len(lq_list)
    num_layers = max(gate_layer.values()) + 1 if gate_layer else 1

    T = np.zeros((n_logical, n_logical, num_layers), dtype=np.float64)
    for gate_id, (q1, q2) in two_qubit_gates.items():
        layer = gate_layer.get(gate_id, 0)
        i, j = lq_index[q1], lq_index[q2]
        T[i][j][layer] += 1.0
        T[j][i][layer] += 1.0

    # Step 3: CP decomposition via ALS
    R = min(8, n_logical, num_layers)
    np.random.seed(42)
    A = np.random.randn(n_logical, R)
    B = np.random.randn(n_logical, R)
    C = np.random.randn(num_layers, R)
    reg = 1e-6

    for _ in range(30):
        # Update A
        T0 = T.reshape(n_logical, -1)
        kr_CB = np.zeros((num_layers * n_logical, R))
        for r in range(R):
            kr_CB[:, r] = np.kron(C[:, r], B[:, r])
        A = T0 @ kr_CB @ np.linalg.inv(kr_CB.T @ kr_CB + reg * np.eye(R))

        # Update B
        T1 = T.transpose(1, 0, 2).reshape(n_logical, -1)
        kr_CA = np.zeros((num_layers * n_logical, R))
        for r in range(R):
            kr_CA[:, r] = np.kron(C[:, r], A[:, r])
        B = T1 @ kr_CA @ np.linalg.inv(kr_CA.T @ kr_CA + reg * np.eye(R))

        # Update C
        T2 = T.transpose(2, 0, 1).reshape(num_layers, -1)
        kr_BA = np.zeros((n_logical * n_logical, R))
        for r in range(R):
            kr_BA[:, r] = np.kron(B[:, r], A[:, r])
        C = T2 @ kr_BA @ np.linalg.inv(kr_BA.T @ kr_BA + reg * np.eye(R))

    # Step 4: Logical qubit embedding from A, B factors
    logical_embedding = np.hstack([A, B])  # (n_logical, 2R)

    # Step 5: Spectral embedding of hardware graph
    phys_qubits = sorted(self.backend.keys())
    phys_index = {q: idx for idx, q in enumerate(phys_qubits)}
    n_phys = len(phys_qubits)

    degree = np.zeros(n_phys)
    adj = np.zeros((n_phys, n_phys))
    for q in phys_qubits:
        for neighbor in self.backend[q]:
            if neighbor in phys_index:
                i_idx, j_idx = phys_index[q], phys_index[neighbor]
                adj[i_idx][j_idx] = 1.0
                degree[i_idx] += 1.0

    laplacian = np.diag(degree) - adj
    embed_dim = min(2 * R, n_phys - 1) if n_phys > 1 else 1
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    if n_phys > embed_dim + 1:
        physical_embedding = eigenvectors[:, 1:embed_dim + 1]
    else:
        physical_embedding = eigenvectors[:, 1:]

    # Step 6: Align dimensions
    log_dim = logical_embedding.shape[1]
    phys_dim = physical_embedding.shape[1]
    if log_dim > phys_dim:
        physical_embedding = np.hstack([physical_embedding, np.zeros((n_phys, log_dim - phys_dim))])
    elif phys_dim > log_dim:
        logical_embedding = np.hstack([logical_embedding, np.zeros((n_logical, phys_dim - log_dim))])

    # Normalize
    for i in range(logical_embedding.shape[0]):
        norm = np.linalg.norm(logical_embedding[i])
        if norm > 1e-10:
            logical_embedding[i] /= norm
    for i in range(physical_embedding.shape[0]):
        norm = np.linalg.norm(physical_embedding[i])
        if norm > 1e-10:
            physical_embedding[i] /= norm

    # Step 7: Hungarian matching on Euclidean distance
    cost_matrix = np.zeros((n_logical, n_phys))
    for i in range(n_logical):
        for j in range(n_phys):
            diff = logical_embedding[i] - physical_embedding[j]
            cost_matrix[i][j] = np.sqrt(np.sum(diff ** 2))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    lq_to_phys = {}
    for r_idx, c_idx in zip(row_ind, col_ind):
        lq_to_phys[lq_list[r_idx]] = phys_qubits[c_idx]

    # Step 8: Build full bijective mapping via swaps from identity
    mapping_dict = list(range(num_qubits))
    reverse_mapping_dict = list(range(num_qubits))

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