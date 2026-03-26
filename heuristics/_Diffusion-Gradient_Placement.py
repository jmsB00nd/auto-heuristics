def init_mapping(self):
    import numpy as np

    n_physical = self.num_qubits  # total physical qubits (0 .. n_physical-1)
    physical_nodes = sorted(self.backend.keys())

    # --- Step 1: Collect logical qubits and build interaction graph W ---
    logical_qubits_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubits_set.add(q)
    logical_qubits = sorted(logical_qubits_set)
    n_logical = len(logical_qubits)
    log_idx = {q: i for i, q in enumerate(logical_qubits)}

    # Interaction weights between logical qubit pairs
    W = np.zeros((n_logical, n_logical))
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            i, j = log_idx[qubits[0]], log_idx[qubits[1]]
            W[i][j] += 1
            W[j][i] += 1

    # --- Step 2: Spectral embedding of hardware graph into 2D ---
    n_hw = len(physical_nodes)
    node_idx = {v: i for i, v in enumerate(physical_nodes)}

    # Build adjacency and degree matrices for graph Laplacian
    A = np.zeros((n_hw, n_hw))
    for v in physical_nodes:
        for u in self.backend[v]:
            A[node_idx[v]][node_idx[u]] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A  # graph Laplacian

    # Use eigenvectors of Laplacian for 2D coordinates
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Skip first eigenvector (constant); take next 2 for 2D embedding
    if n_hw >= 3:
        hw_coords = eigenvectors[:, 1:3]
    elif n_hw == 2:
        hw_coords = eigenvectors[:, 1:2]
        hw_coords = np.hstack([hw_coords, np.zeros((n_hw, 1))])
    else:
        hw_coords = np.zeros((n_hw, 2))

    # Physical qubit index -> 2D coordinate
    phys_coords = {physical_nodes[i]: hw_coords[i] for i in range(n_hw)}

    # --- Step 3: Compute centroid (min sum-of-distances node) ---
    min_sum = float('inf')
    centroid = physical_nodes[0]
    for v in physical_nodes:
        s = sum(self.distance_matrix[v][u] for u in physical_nodes
                if self.distance_matrix[v][u] < float('inf'))
        if s < min_sum:
            min_sum = s
            centroid = v

    centroid_coord = phys_coords[centroid]

    # --- Step 4: Initialize all logical qubits at centroid, then simulate forces ---
    positions = np.array([centroid_coord.copy() for _ in range(n_logical)])

    # Add tiny random perturbation to break symmetry
    rng = np.random.RandomState(42)
    positions += rng.randn(n_logical, 2) * 1e-4

    # Force simulation parameters
    dt = 0.05
    damping = 0.9
    n_iterations = 150
    velocities = np.zeros((n_logical, 2))

    max_w = W.max() if W.max() > 0 else 1.0

    for _ in range(n_iterations):
        forces = np.zeros((n_logical, 2))
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    diff = rng.randn(2) * 1e-4
                    dist = np.linalg.norm(diff)
                direction = diff / dist

                w_ij = W[i][j]

                # Repulsive force: inversely proportional to interaction weight
                # High interaction -> low repulsion (they want to stay close)
                repulsion_coeff = 1.0 / (1.0 + w_ij)
                repulsive = repulsion_coeff / (dist * dist + 1e-6) * direction

                # Attractive force: proportional to interaction weight
                # High interaction -> strong attraction
                attractive = -w_ij / max_w * dist * 0.1 * direction

                force = repulsive + attractive
                forces[i] += force
                forces[j] -= force

        velocities = damping * (velocities + forces * dt)
        positions += velocities * dt

    # --- Step 5: Greedy snap to nearest unoccupied physical qubit ---
    assignments = []
    for i in range(n_logical):
        dists = []
        for v in physical_nodes:
            d = np.linalg.norm(positions[i] - phys_coords[v])
            dists.append((d, v))
        dists.sort()
        assignments.append((i, dists))

    # Sort by closest distance to best target (greediest first)
    assignments.sort(key=lambda x: x[1][0][0])

    occupied = set()
    mapping_dict_list = list(range(n_physical))
    reverse_mapping_dict_list = list(range(n_physical))
    assigned_logical = set()

    for idx, dists in assignments:
        lq = logical_qubits[idx]
        for d, pq in dists:
            if pq not in occupied:
                mapping_dict_list[lq] = pq
                reverse_mapping_dict_list[pq] = lq
                occupied.add(pq)
                assigned_logical.add(lq)
                break

    # Assign remaining logical qubits (if any unused) to remaining physical qubits
    remaining_physical = [p for p in range(n_physical) if p not in occupied]
    remaining_logical = [q for q in range(n_physical) if q not in assigned_logical]
    for lq, pq in zip(remaining_logical, remaining_physical):
        mapping_dict_list[lq] = pq
        reverse_mapping_dict_list[pq] = lq

    # --- Step 6: Populate mapping ---
    self.mapping_dict = mapping_dict_list
    self.reverse_mapping_dict = reverse_mapping_dict_list

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)