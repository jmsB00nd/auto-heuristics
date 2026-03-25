def init_mapping(self):
    """Gravity-Well Placement: physics-based force simulation for initial mapping."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    # ── 1. Collect logical qubits and build interaction graph ────────────
    logical_qubits = sorted({q for qubits in self.access.values() for q in qubits})
    num_logical = len(logical_qubits)

    if num_logical == 0:
        self.mapping_dict = {}
        self.reverse_mapping_dict = {}
        return

    log_idx = {q: i for i, q in enumerate(logical_qubits)}

    # Interaction weights between logical qubit pairs and per-qubit mass
    interaction_weights = defaultdict(float)
    qubit_mass = defaultdict(float)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            edge = (min(q1, q2), max(q1, q2))
            interaction_weights[edge] += 1.0
            qubit_mass[q1] += 1.0
            qubit_mass[q2] += 1.0

    # ── 2. Physical qubits and hardware centrality ───────────────────────
    phys_qubits = sorted(self.backend.keys())
    num_phys = len(phys_qubits)

    # Closeness centrality from distance_matrix
    centrality = np.zeros(num_phys)
    for i, p in enumerate(phys_qubits):
        total_dist = sum(
            self.distance_matrix[p][q] for q in phys_qubits
            if q != p and self.distance_matrix[p][q] < float('inf')
        )
        centrality[i] = (num_phys - 1) / total_dist if total_dist > 0 else 0.0

    # ── 3. Spectral embedding of hardware graph (MDS on distance matrix) ─
    # Classical MDS: double-center the squared distance matrix, eigendecompose
    k = 2
    D = np.zeros((num_phys, num_phys))
    for i, p1 in enumerate(phys_qubits):
        for j, p2 in enumerate(phys_qubits):
            d = self.distance_matrix[p1][p2]
            D[i, j] = d if d < float('inf') else 0.0

    D_sq = D ** 2
    H_mat = np.eye(num_phys) - np.ones((num_phys, num_phys)) / num_phys
    B = -0.5 * H_mat @ D_sq @ H_mat

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx_sorted = np.argsort(eigenvalues)[::-1][:k]
        phys_coords = eigenvectors[:, idx_sorted] * np.sqrt(
            np.maximum(eigenvalues[idx_sorted], 0.0)
        )
    except np.linalg.LinAlgError:
        phys_coords = np.random.randn(num_phys, k) * 0.1

    # ── 4. Initialize logical qubit positions ────────────────────────────
    # Heavy logical qubits start near high-centrality physical positions;
    # all positions receive random perturbation to break symmetry.
    c_sum = centrality.sum()
    if c_sum > 0:
        centrality_weighted_center = (centrality[:, None] * phys_coords).sum(axis=0) / c_sum
    else:
        centrality_weighted_center = phys_coords.mean(axis=0)

    spread = max(phys_coords.std(), 1e-3)
    positions = np.zeros((num_logical, k))
    rng = np.random.default_rng(42)
    for i, q in enumerate(logical_qubits):
        mass = qubit_mass.get(q, 0.0) + 1.0
        # Heavier qubits start closer to centrality-weighted center
        alpha = min(mass / (mass + 5.0), 0.9)
        positions[i] = (
            alpha * centrality_weighted_center
            + (1.0 - alpha) * phys_coords[rng.integers(num_phys)]
            + rng.normal(0, spread * 0.3, size=k)
        )

    # ── 5. Damped force simulation ───────────────────────────────────────
    dt = 0.05
    damping = 0.85
    velocities = np.zeros((num_logical, k))
    max_iters = 150

    # Precompute interaction list for efficiency
    interactions = [
        (log_idx[q1], log_idx[q2], w)
        for (q1, q2), w in interaction_weights.items()
    ]
    masses = np.array([qubit_mass.get(q, 0.0) + 1.0 for q in logical_qubits])

    for iteration in range(max_iters):
        forces = np.zeros((num_logical, k))

        # (a) Attractive forces between interacting logical qubit pairs
        #     Spring-like: proportional to gate weight, capped to prevent blowup
        for i, j, w in interactions:
            diff = positions[j] - positions[i]
            dist = np.linalg.norm(diff) + 1e-10
            f_attract = w * diff / dist * min(dist, 2.0 * spread)
            forces[i] += f_attract
            forces[j] -= f_attract

        # (b) Repulsive forces between all logical qubit pairs (Coulomb-like)
        #     Prevents overlap: inversely proportional to distance squared
        for i in range(num_logical):
            for j in range(i + 1, num_logical):
                diff = positions[j] - positions[i]
                dist_sq = np.dot(diff, diff) + 1e-10
                dist = np.sqrt(dist_sq)
                repulsion = spread * diff / (dist_sq * dist) * 0.5
                forces[i] -= repulsion
                forces[j] += repulsion

        # (c) Gravitational pull toward high-centrality hardware nodes
        #     Scales with logical qubit mass × physical centrality / distance²
        for i in range(num_logical):
            grav = np.zeros(k)
            m_i = masses[i]
            for j_p in range(num_phys):
                diff = phys_coords[j_p] - positions[i]
                dist = np.linalg.norm(diff) + 1e-10
                grav += centrality[j_p] * m_i * diff / (dist ** 2)
            forces[i] += grav * 0.02

        # Update velocities (damped) and positions
        velocities = damping * (velocities + forces * dt)
        positions += velocities * dt

        # Early convergence check
        if np.max(np.abs(velocities)) < 1e-7:
            break

    # ── 6. Snap to physical qubits via Hungarian algorithm ───────────────
    # Cost = Euclidean distance between converged logical positions and
    # physical qubit coordinates in the MDS embedding space.
    # linear_sum_assignment guarantees a strict 1-to-1 matching.
    cost = np.zeros((num_logical, num_phys))
    for i in range(num_logical):
        for j in range(num_phys):
            cost[i, j] = np.linalg.norm(positions[i] - phys_coords[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    self.mapping_dict = {}
    self.reverse_mapping_dict = {}
    for li, pi in zip(row_ind, col_ind):
        lq = logical_qubits[li]
        pq = phys_qubits[pi]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)