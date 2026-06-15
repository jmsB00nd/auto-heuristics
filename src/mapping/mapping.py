import random
from collections import defaultdict, deque
import re
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag


from mqt.core import load
from mqt.qmap.sc import (
    Architecture as QMapArchitecture, map_, Configuration,
    Method, Heuristic, InitialLayout, Layering, EarlyTermination,
    LookaheadHeuristic, Encoding, CommanderGrouping, SwapReduction,
)

from pytket.qasm import circuit_from_qasm_str
from pytket.architecture import Architecture as TketArchitecture
from pytket.placement import GraphPlacement, LinePlacement

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

def generate_random_initial_mapping(num_qubits: int):
    """Random shuffle: logical i -> physical perm(i)."""
    logical_qubits = list(range(num_qubits))
    physical_qubits = list(range(num_qubits))
    random.shuffle(physical_qubits)

    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits
    for logical_qubit, physical_qubit in zip(logical_qubits, physical_qubits):
        mapping[logical_qubit] = physical_qubit
        reverse_mapping[physical_qubit] = logical_qubit
    return mapping, reverse_mapping


def generate_trivial_initial_mapping(num_qubits: int):
    """Identity: logical i -> physical i."""
    mapping = list(range(num_qubits))
    reverse_mapping = list(range(num_qubits))
    return mapping, reverse_mapping


def generate_qmap_initial_mapping(qasm_code, backend_edges, num_qubits):
    circuit = QuantumCircuit.from_qasm_str(qasm_code)

    # qmap rejects non-unitary ops (e.g. reset); strip them before mapping.
    # Only the initial layout is needed here, so dropping them is safe.
    if any(instr.name == 'reset' for instr, _, _ in circuit.data):
        stripped = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        for instr, qargs, cargs in circuit.data:
            if instr.name != 'reset':
                stripped.append(instr, qargs, cargs)
        circuit = stripped

    edges = {tuple(e) for e in CouplingMap(backend_edges).get_edges()}
    max_q = max(max(u, v) for u, v in edges) + 1

    arch = QMapArchitecture()
    arch.num_qubits = max_q
    arch.coupling_map = edges


    config = Configuration()
    config.method = Method.heuristic
    config.heuristic = Heuristic.gate_count_max_distance
    config.initial_layout = InitialLayout.dynamic
    config.iterative_bidirectional_routing = False
    config.layering = Layering.individual_gates
    config.automatic_layer_splits = True
    config.automatic_layer_splits_node_limit = 5000
    config.early_termination = EarlyTermination.none
    config.early_termination_limit = 0
    config.encoding = Encoding.commander
    config.commander_grouping = CommanderGrouping.fixed3
    config.swap_reduction = SwapReduction.coupling_limit
    config.swap_limit = 0
    config.include_wcnf = False
    config.use_subsets = True
    config.subgraph = set()
    config.pre_mapping_optimizations = True
    config.post_mapping_optimizations = False
    config.add_measurements_to_mapped_circuit = False
    config.add_barriers_between_layers = False
    config.lookahead_heuristic = LookaheadHeuristic.gate_count_max_distance
    config.lookaheads = 15
    config.lookahead_factor = 0.5

    qc_mapped, _ = map_(load(circuit), arch, config)

    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for physical, logical in sorted(qc_mapped.initial_layout.items()):
        if 0 <= logical < num_qubits and 0 <= physical < num_qubits \
                and mapping[logical] == -1 and reverse_mapping[physical] == -1:
            mapping[logical] = physical
            reverse_mapping[physical] = logical

    free_physical = [p for p in range(num_qubits) if reverse_mapping[p] == -1]
    for logical in range(num_qubits):
        if mapping[logical] == -1 and free_physical:
            physical = free_physical.pop(0)
            mapping[logical] = physical
            reverse_mapping[physical] = logical

    return mapping, reverse_mapping

def generate_pytket_initial_mapping(qasm_code, backend_edges, num_qubits,
                                    graph_timeout_ms=5000):
    """
    Use pytket's GraphPlacement to generate an initial layout.

    GraphPlacement performs a weighted subgraph-monomorphism search that can
    blow past its internal time/match budget on large or dense circuits,
    raising "GraphPlacement execution time has exceeded allowed limits."
    When that happens we fall back to the (fast) LinePlacement, and finally to
    a trivial identity layout, so a single hard circuit never aborts the run.

    Returns:
    - mapping[logical] = physical
    - reverse_mapping[physical] = logical
    """

    circuit = circuit_from_qasm_str(qasm_code)

    # Architecture needs a sequence, not a set
    edges = list(backend_edges)

    # If your edges are flat integer pairs, this is enough:
    architecture = TketArchitecture(edges)

    placement_map = None
    for placer in (GraphPlacement(architecture, timeout=graph_timeout_ms),
                   LinePlacement(architecture)):
        try:
            placement_map = placer.get_placement_map(circuit)
            break
        except Exception:
            # Try the next (cheaper) placer.
            continue

    if placement_map is None:
        # Both placers gave up — trivial identity is always valid.
        return generate_trivial_initial_mapping(num_qubits)

    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for logical_qb, physical_node in placement_map.items():
        if logical_qb.reg_name == "ancilla":
            continue

        # Qubits pytket could not place go to a virtual "unplaced" register
        # whose indices restart at 0 and would collide with real "node"
        # indices. Skip them and let the fill step below assign free qubits.
        if physical_node.reg_name != "node":
            continue

        logical_idx = logical_qb.index[0]
        physical_idx = physical_node.index[0]

        if 0 <= logical_idx < num_qubits and 0 <= physical_idx < num_qubits:
            mapping[logical_idx] = physical_idx
            reverse_mapping[physical_idx] = logical_idx

    # Placement only assigns qubits involved in interactions; fill the rest
    # with free physical qubits so the mapping is a complete permutation.
    free_physical = [p for p in range(num_qubits) if reverse_mapping[p] == -1]
    for logical in range(num_qubits):
        if mapping[logical] == -1 and free_physical:
            physical = free_physical.pop(0)
            mapping[logical] = physical
            reverse_mapping[physical] = logical

    return mapping, reverse_mapping

def generate_cirq_initial_mapping(qasm_code, backend_edges, num_qubits):
    """
    Use Cirq's RouteCQC transformer to generate an initial layout.
    Returned as arrays:
    - mapping[logical] = physical
    - reverse_mapping[physical] = logical
    """
    circuit = circuit_from_qasm(qasm_code)
    
    device_graph = nx.Graph()
    for source, target in backend_edges:
        device_graph.add_edge(cirq.LineQubit(source), cirq.LineQubit(target))
        
    router = cirq.RouteCQC(device_graph)
    
    routed_circuit, initial_map, swap_map = router.route_circuit(circuit)
    
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits
    
    for logical_q, physical_q in initial_map.items():
        if "ancilla" in str(logical_q).lower():
            continue

        physical_idx = physical_q.x
        
        if isinstance(logical_q, cirq.LineQubit):
            logical_idx = logical_q.x
        else:
            match = re.search(r'\d+', str(logical_q))
            if match:
                logical_idx = int(match.group())
            else:
                continue  # Skip if we can't parse a valid index
                
        if 0 <= logical_idx < num_qubits and 0 <= physical_idx < num_qubits:
            mapping[logical_idx] = physical_idx
            reverse_mapping[physical_idx] = logical_idx
            
    return mapping, reverse_mapping

def generate_sabre_initial_mapping(qasm_code, backend_edges, num_qubits):
    """Qiskit's SabreLayout (iterative routing-driven layout refinement)."""
    circuit = QuantumCircuit.from_qasm_str(qasm_code)
    dag_circuit = circuit_to_dag(circuit)
    coupling_map = CouplingMap(backend_edges)
    sabre_layout = SabreLayout(coupling_map, seed=21)
    sabre_layout.run(dag_circuit)

    layout = sabre_layout.property_set["layout"]
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for v in layout._v2p:
        if v._register.name == "ancilla":
            continue
        logical_idx = v._index
        physical_idx = layout._v2p[v]
        if logical_idx < num_qubits and physical_idx < num_qubits:
            mapping[logical_idx] = physical_idx
            reverse_mapping[physical_idx] = logical_idx
    return mapping, reverse_mapping


def _interaction_graph(access):
    """Build the weighted logical interaction graph from a schedule.

    Returns (weights, neighbors) where weights[(u,v)] (u<v) is the count
    of 2q gates between u and v, and neighbors[q] is the set of logical
    qubits q interacts with.
    """
    weights = defaultdict(float)
    neighbors = defaultdict(set)
    for qs in access.values():
        if len(qs) != 2 or qs[0] == qs[1]:
            continue
        u, v = qs
        key = (u, v) if u < v else (v, u)
        weights[key] += 1.0
        neighbors[u].add(v)
        neighbors[v].add(u)
    return weights, neighbors


def _spectral_coembed(weights, neighbors, active, backend, phys_size,
                      num_qubits, ndim=2):
    """Spectral co-embedding of the logical and hardware graphs.

    Both the weighted logical interaction graph and the (unweighted)
    hardware graph are embedded into a common ``ndim``-dimensional
    Euclidean space using the low-frequency eigenvectors of their
    respective graph Laplacians (the classic spectral graph-drawing
    construction). The two point clouds are each standardised per
    axis to a common scale, and logical qubits are matched to
    physical qubits by a single optimal-assignment (Hungarian) solve
    that minimises total squared embedding distance.

    Rationale: a Laplacian eigen-embedding lays out a graph so that
    strongly-interacting vertices sit close together, recovering the
    graph's intrinsic geometry. When the logical interaction graph is
    *mesh-like* — denser than the device yet with genuine 2D extent —
    aligning its spectral geometry to the device's own spectral
    geometry places heavy clusters of interactions onto compact, well-
    connected hardware regions. This is a global, deterministic
    placement that a local routing-driven layout (Sabre) does not
    perform. The result is a single mapping; no candidate enumeration.

    Returns (mapping, reverse_mapping).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def _laplacian_embed(nodes, adj, wfun, k):
        idx = {nd: i for i, nd in enumerate(nodes)}
        N = len(nodes)
        L = np.zeros((N, N))
        for u in nodes:
            iu = idx[u]
            for v in adj[u]:
                if v in idx:
                    w = wfun(u, v)
                    L[iu][idx[v]] -= w
                    L[iu][iu] += w
        # Symmetric Laplacian -> real eigh; deterministic for fixed L.
        _, vecs = np.linalg.eigh(L)
        # Skip the trivial constant eigenvector (index 0); take the
        # next k low-frequency modes as coordinates.
        k = min(k, max(1, N - 1))
        return idx, vecs[:, 1:1 + k]

    li, lvec = _laplacian_embed(
        active, neighbors,
        lambda u, v: weights[(u, v) if u < v else (v, u)], ndim)
    hw_nodes = [p for p in backend if p < phys_size]
    hi, hvec = _laplacian_embed(hw_nodes, backend, lambda u, v: 1.0, ndim)

    # Standardise each coordinate axis (zero mean, unit variance) so the
    # two embeddings share a comparable scale before matching.
    def _standardise(M):
        mu = M.mean(axis=0)
        sd = M.std(axis=0) + 1e-9
        return (M - mu) / sd
    L = _standardise(lvec)
    H = _standardise(hvec)

    # Cost matrix: squared Euclidean distance between every logical
    # point and every physical point; optimal one-to-one assignment.
    cost = np.zeros((len(active), len(hw_nodes)))
    for i, q in enumerate(active):
        diff = H - L[li[q]]
        cost[i] = np.einsum('ij,ij->i', diff, diff)
    rows, cols = linear_sum_assignment(cost)

    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits
    used = set()
    for i, j in zip(rows, cols):
        lq = active[i]
        pq = hw_nodes[j]
        if 0 <= lq < num_qubits and 0 <= pq < phys_size:
            mapping[lq] = pq
            reverse_mapping[pq] = lq
            used.add(pq)
    for lq in range(num_qubits):
        if mapping[lq] != -1:
            continue
        for p in range(phys_size):
            if p not in used:
                mapping[lq] = p
                reverse_mapping[p] = lq
                used.add(p)
                break
    return mapping, reverse_mapping


def _subgraph_embed(neighbors, weights, active, backend, phys_size,
                    step_budget):
    """Deterministic subgraph-monomorphism embedding.

    Search for an injective map ``f : V_logical -> V_hardware`` such
    that every logical interaction edge ``(u, v)`` maps to a hardware
    edge ``(f(u), f(v))``. If such an ``f`` exists the routed circuit
    needs *zero* SWAPs, since every two-qubit gate is already on a
    coupled physical pair.

    This is the exact structure the QUEKO-style benchmarks are built
    around (a perfect 0-SWAP layout exists by construction), yet a
    randomised routing-driven layout such as Sabre does not always
    recover it. A bounded VF2-style backtracking search recovers it
    deterministically:

    * Logical qubits are ordered by a BFS from the maximum-degree
      vertex, so each newly assigned qubit already has placed
      neighbours — every candidate is then constrained to the
      intersection of those neighbours' hardware adjacencies, which
      prunes the search hard.
    * A candidate physical qubit must have hardware degree at least
      the logical degree (necessary condition), giving an immediate
      fail-fast on dense/hub graphs that cannot embed.
    * ``step_budget`` bounds total recursive expansions so the search
      stays cheap; on graphs that do not embed it returns ``None``
      quickly and the caller falls back to another strategy.

    Returns the logical->physical dict on success, else ``None``.
    Heaviest-weight edges are explored first inside each candidate
    set so the returned embedding is deterministic.
    """
    active_set = set(active)
    deg = {q: len(neighbors[q]) for q in active}
    hw_deg = {p: len(backend[p]) for p in backend}

    # BFS ordering from the most-constrained (highest-degree) vertex,
    # heaviest neighbours first — fully deterministic.
    root = max(active, key=lambda q: (deg[q], -q))
    order = []
    seen = {root}
    dq = deque([root])
    while dq:
        x = dq.popleft()
        order.append(x)
        nbrs = sorted(
            (y for y in neighbors[x] if y in active_set and y not in seen),
            key=lambda z: (-weights[(x, z) if x < z else (z, x)], -deg[z], z))
        for y in nbrs:
            seen.add(y)
            dq.append(y)
    for q in active:
        if q not in seen:
            order.append(q)
            seen.add(q)

    log_to_phys = {}
    phys_used = set()
    steps = [0]

    hw_nodes_by_deg = sorted(
        (p for p in backend if p < phys_size),
        key=lambda p: (-hw_deg[p], p))

    import sys as _sys
    old_limit = _sys.getrecursionlimit()
    _sys.setrecursionlimit(max(old_limit, len(order) + 100))

    def backtrack(idx):
        if steps[0] > step_budget:
            return False
        steps[0] += 1
        if idx == len(order):
            return True
        q = order[idx]
        placed_nbrs = [nb for nb in neighbors[q] if nb in log_to_phys]
        if placed_nbrs:
            cand = None
            for nb in placed_nbrs:
                p_nb = log_to_phys[nb]
                nbset = backend[p_nb]
                if cand is None:
                    cand = set(nbset)
                else:
                    cand &= nbset
                if not cand:
                    break
            cand = cand or set()
            candidates = sorted(
                c for c in cand
                if c not in phys_used and c < phys_size
                and hw_deg.get(c, 0) >= deg[q])
        else:
            candidates = [p for p in hw_nodes_by_deg
                          if p not in phys_used and hw_deg[p] >= deg[q]]
        for p in candidates:
            log_to_phys[q] = p
            phys_used.add(p)
            if backtrack(idx + 1):
                return True
            del log_to_phys[q]
            phys_used.discard(p)
        return False

    try:
        ok = backtrack(0)
    finally:
        _sys.setrecursionlimit(old_limit)

    return log_to_phys if ok else None


def _logical_bfs_layers(neighbors, weights, active, root):
    """BFS-layer the logical interaction graph from `root`.

    Returns (layer_of, layers) where layer_of[q] = bfs depth from root,
    and layers[d] = list of logical qubits at depth d (deterministic order:
    qubits with larger weighted edge into already-laid layer come first).
    """
    layer_of = {root: 0}
    layers = [[root]]
    frontier = [root]
    depth = 0
    while frontier:
        next_frontier_set = set()
        candidates_w = {}
        for u in frontier:
            for v in neighbors[u]:
                if v in layer_of or v not in active:
                    continue
                next_frontier_set.add(v)
                w = weights[(u, v) if u < v else (v, u)]
                candidates_w[v] = candidates_w.get(v, 0.0) + w
        if not next_frontier_set:
            break
        depth += 1
        # Order: heaviest pull into current frontier first, then by id.
        ordered = sorted(next_frontier_set, key=lambda q: (-candidates_w[q], q))
        for q in ordered:
            layer_of[q] = depth
        layers.append(ordered)
        frontier = ordered
    return layer_of, layers


def _hardware_bfs_from(backend, start, phys_size):
    """BFS distance dictionary from `start` on the hardware graph."""
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in backend[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _pick_hardware_root(backend, distance_matrix, phys_size):
    """Pick the hardware root: maximum-degree physical qubit with the
    smallest eccentricity (graph-center bias), tie-broken by smaller id.
    """
    nodes = [p for p in backend.keys() if p < phys_size]
    if not nodes:
        nodes = list(backend.keys())

    def ecc(p):
        d = _hardware_bfs_from(backend, p, phys_size)
        return max(d.values()) if d else 0

    best = None
    best_key = None
    for p in nodes:
        deg = len(backend[p])
        e = ecc(p)
        key = (-deg, e, p)
        if best_key is None or key < best_key:
            best_key = key
            best = p
    return best


def _temporal_weights(access):
    """Weighted logical interaction graph with an earliness bonus.

    Each 2q gate of schedule rank r (0-based over the M 2q gates, in
    schedule order) contributes ``1 + (M - r) / M`` to its pair's weight:
    the raw gate count dominates, while earlier gates count up to twice
    as much as the latest ones. The initial mapping only directly serves
    the early part of the circuit (later on, routing has already moved
    qubits), so ties between equally-frequent pairs must break toward
    the pairs that interact first.

    Returns (W, neighbors, first_use) where W[(u,v)] (u<v) is the
    temporal weight, neighbors[q] the interaction set, and first_use[q]
    the rank of q's first 2q gate (for deterministic orderings).
    """
    pair_gates = []
    for key in sorted(access.keys()):
        qs = access[key]
        if len(qs) == 2 and qs[0] != qs[1]:
            pair_gates.append(tuple(qs))
    W = defaultdict(float)
    neighbors = defaultdict(set)
    first_use = {}
    touches = defaultdict(int)
    for r, (u, v) in enumerate(pair_gates):
        key = (u, v) if u < v else (v, u)
        # influence decay: every earlier gate touching u or v gave the
        # router an opportunity to have already moved them, so this
        # gate constrains the *initial* mapping proportionally less.
        W[key] += 1.0 / (1 + touches[u] + touches[v])
        touches[u] += 1
        touches[v] += 1
        neighbors[u].add(v)
        neighbors[v].add(u)
        if u not in first_use:
            first_use[u] = r
        if v not in first_use:
            first_use[v] = r
    return W, neighbors, first_use


def _long_hardware_path(backend, phys_size):
    """Deterministic long simple path through the hardware graph.

    Greedy wall-following DFS tried from every start node: repeatedly
    step to the unvisited neighbour with the fewest unvisited
    neighbours (ties by id). Returns the longest path found. O(V^2).
    """
    best = []
    for start in sorted(backend.keys()):
        if start >= phys_size:
            continue
        path = [start]
        used = {start}
        cur = start
        while True:
            cands = [n for n in backend[cur]
                     if n < phys_size and n not in used]
            if not cands:
                break
            def _step_key(x):
                free = sum(1 for y in backend[x]
                           if y < phys_size and y not in used)
                # Warnsdorff with dead-end avoidance: hug the wall
                # (fewest onward options first) but never walk into a
                # dead end while a live continuation exists — hanging
                # stubs (heavy-hex bridge qubits) would otherwise
                # truncate the path.
                return (free == 0, free, x)
            nxt = min(cands, key=_step_key)
            path.append(nxt)
            used.add(nxt)
            cur = nxt
        if len(path) > len(best):
            best = path
        if len(best) >= phys_size:
            break
    return best


def _face_region(backend, phys_size, hw_root, n_target, distance_matrix):
    """Grow a placement region anchored on a closed hardware cycle.

    A cluster of interacting qubits routes well only if its occupied
    region contains closed cycles: a cyclic region lets the router
    rotate tokens (carousel) without stranding them, while a tree
    region — which pure distance minimisation happily produces on
    sparse devices — bottlenecks every move. We therefore anchor the
    region on the shortest cycle through the hardware root and grow it
    by always preferring the free site with the most region
    neighbours (closing further cycles before sprouting stubs).

    Returns a list of sites of size >= max(n_target, |face|) (face
    kept whole so one slack carousel slot survives even when
    n_target is smaller), or None when the graph has no cycle.
    """
    # shortest cycle through hw_root: for each pair of root neighbours,
    # BFS distance between them in G - root.
    nbrs_root = [n for n in backend[hw_root] if n < phys_size]
    best_cycle = None
    for i in range(len(nbrs_root)):
        for j in range(i + 1, len(nbrs_root)):
            a, b = nbrs_root[i], nbrs_root[j]
            # BFS from a to b avoiding hw_root
            prev = {a: None}
            dq = deque([a])
            while dq:
                x = dq.popleft()
                if x == b:
                    break
                for y in backend[x]:
                    if y < phys_size and y != hw_root and y not in prev:
                        prev[y] = x
                        dq.append(y)
            if b not in prev:
                continue
            path = [b]
            while path[-1] is not None:
                path.append(prev[path[-1]])
            path.pop()
            cycle = [hw_root] + path[::-1]
            if best_cycle is None or len(cycle) < len(best_cycle):
                best_cycle = cycle
    if best_cycle is None:
        return None
    region = list(best_cycle)
    rset = set(region)
    target = max(n_target, len(region))
    while len(region) < target:
        cands = set()
        for s in region:
            for a in backend[s]:
                if a < phys_size and a not in rset:
                    cands.add(a)
        if not cands:
            break
        def grow_key(c):
            closing = sum(1 for y in backend[c] if y in rset)
            dface = min(distance_matrix[c][f] for f in best_cycle)
            return (-closing, dface, c)
        nxt = min(cands, key=grow_key)
        region.append(nxt)
        rset.add(nxt)
    return region


def _best_path_offset(order, snake, edge_w, distance_matrix,
                      tie='centre'):
    """Slide the qubit sequence `order` along the hardware path `snake`
    and return the offset minimising the weighted stretch
    ``sum_e w_e * (dist - 1)`` of the given edges.

    Ties break per regime: ``'centre'`` leaves router slack on both
    ends (best for a walking hub serving a chain of one-shot blocks);
    ``'corner'`` anchors the segment at the path start, where the
    wall-following construction packs consecutive slots into a compact
    block (best when every qubit participates in the sweep).
    """
    P, L = len(snake), len(order)
    if L > P:
        return 0
    idx = {q: i for i, q in enumerate(order)}
    centre = (P - L) / 2.0
    best = None
    for off in range(P - L + 1):
        c = 0.0
        for (u, v), w in edge_w.items():
            iu, iv = idx.get(u), idx.get(v)
            if iu is None or iv is None:
                continue
            d = distance_matrix[snake[off + iu]][snake[off + iv]]
            if d == float('inf'):
                d = P + 1
            c += w * (d - 1)
        tiekey = abs(off - centre) if tie == 'centre' else off
        key = (c, tiekey, off)
        if best is None or key < best[0]:
            best = (key, off)
    return best[1]


def generate_gem_initial_mapping(qasm_code, backend_edges, num_qubits,
                                 access, distance_matrix, backend):
    """GEM — Graph Embedding Mapping with hub-trail transformation.

    Pipeline (single pass, deterministic, parameter-free):

    1. **Exact embedding.** If the logical interaction graph is a
       subgraph of the hardware graph, return that embedding: every 2q
       gate is already on a coupled pair, so the routed SWAP count is
       zero by construction.

    2. **Hub-trail transformation.** A *transient giant hub* — a qubit
       that interacts with essentially the whole register, each pair
       only within a tiny time window — can never have all partners
       adjacent (its logical degree exceeds any hardware degree), so
       the router must walk it across the device. What the walk needs
       is not hub-partner proximity but *consecutive partners* being
       near each other. We therefore replace every such hub gate by a
       virtual gate between the hub's consecutive distinct partners
       (the "trail"), turning the temporal visit sequence into spatial
       adjacency demands the placement stages below can satisfy.

    3. **Exact embedding of the transformed graph.** Trails of simple
       visit patterns (BV stars, KNN/swap-test block scans) are paths
       or ladders that embed exactly; the hub is then seated next to
       its first partner and walks a zero-stretch corridor.

    4. **Cluster placement + influence-decayed exchange descent.**
       Otherwise place qubits greedily by maximum weighted attachment
       and refine with a deterministic pairwise-exchange descent on
       ``sum_e W[e] * (dist - 1)``, where each gate's weight decays
       with the number of earlier gates touching its qubits (every
       earlier touch is a routing opportunity to have moved them, so
       late gates constrain the *initial* mapping less).
    """
    weights, neighbors = _interaction_graph(access)
    if not weights:
        return generate_trivial_initial_mapping(num_qubits)

    active = sorted(neighbors.keys())
    phys_size = len(distance_matrix)
    hw_max_deg = max(len(backend[p]) for p in backend)

    def wkey(u, v):
        return (u, v) if u < v else (v, u)

    BIG = phys_size + 1

    def dist(p, q):
        d = distance_matrix[p][q]
        return BIG if d == float('inf') else d

    def _finalize(log_to_phys):
        new_map = [-1] * num_qubits
        new_rev = [-1] * num_qubits
        used = set()
        for lq, pq in log_to_phys.items():
            if 0 <= lq < num_qubits and 0 <= pq < phys_size:
                new_map[lq] = pq
                new_rev[pq] = lq
                used.add(pq)
        for lq in range(num_qubits):
            if new_map[lq] != -1:
                continue
            for p in range(phys_size):
                if p not in used:
                    new_map[lq] = p
                    new_rev[p] = lq
                    used.add(p)
                    break
        return new_map, new_rev

    def try_embed(nbrs_d, w_d, act):
        # degree pre-check: embedding is impossible if any logical
        # degree exceeds the maximum hardware degree.
        if any(len(nbrs_d[q]) > hw_max_deg for q in act):
            return None
        return _subgraph_embed(nbrs_d, w_d, act, backend, phys_size, 200000)

    # ---- Stage 1: exact zero-SWAP embedding --------------------------------
    embed = try_embed(neighbors, weights, active)
    if embed is not None:
        return _finalize(embed)

    # ---- Stage 2: hub-trail transformation ---------------------------------
    pair_gates = []
    for key in sorted(access.keys()):
        qs = access[key]
        if len(qs) == 2 and qs[0] != qs[1]:
            pair_gates.append(tuple(qs))
    M2 = len(pair_gates)

    # per-pair activity spans
    spans = {}
    for r, (u, v) in enumerate(pair_gates):
        k = wkey(u, v)
        if k in spans:
            spans[k][1] = r
            spans[k][2] += 1
        else:
            spans[k] = [r, r, 1]

    def hub_transience(h):
        num = den = 0.0
        for v in neighbors[h]:
            s = spans[wkey(h, v)]
            num += s[2] * ((s[1] - s[0]) / max(M2 - 1, 1))
            den += s[2]
        return num / den if den else 0.0

    giants = {q for q in active
              if len(neighbors[q]) >= max(hw_max_deg + 1,
                                          0.75 * (len(active) - 1))
              and hub_transience(q) < 0.1}

    if len(giants) >= 0.75 * len(active):
        # Hub-dominated sweep (e.g. QFT): every qubit visits every
        # other in generational order; the trail graph degenerates to a
        # dense band. The right layout is the corridor itself: all
        # qubits along the device's long path in first-use order,
        # centred so the router has slack at both ends.
        first_use0 = {}
        for r, (u, v) in enumerate(pair_gates):
            if u not in first_use0:
                first_use0[u] = r
            if v not in first_use0:
                first_use0[v] = r
        order = sorted(active, key=lambda q: (first_use0.get(q, 0), q))
        snake = _long_hardware_path(backend, phys_size)
        pos = {}
        if len(order) <= len(snake):
            # score offsets with the trail-band weights so the segment
            # lands on a straight stretch of the device path
            trail_w = defaultdict(float)
            lastp = {}
            for (u, v) in pair_gates:
                for h, p in ((u, v), (v, u)):
                    prev = lastp.get(h)
                    if prev is not None and prev != p:
                        trail_w[wkey(prev, p)] += 1.0
                    lastp[h] = p
            off = _best_path_offset(order, snake, trail_w,
                                    distance_matrix, tie='corner')
            for i, q in enumerate(order):
                pos[q] = snake[off + i]
            return _finalize(pos)
        giants = set()           # fall through to cluster placement

    if giants:
        # Pure-star special case: if every non-giant qubit interacts
        # *only* with giants (BV-style fan-out), there are no residual
        # adjacency constraints at all — the entire problem is the
        # giant's walk. Virtual-execution placement handles that best:
        # each leaf is seated next to the giant's *drifted* position at
        # the moment of its gate, so the walk stays maximally fed.
        if all(neighbors[q] <= giants for q in active if q not in giants):
            return generate_vex_initial_mapping(
                qasm_code, backend_edges, num_qubits, access,
                distance_matrix, backend)
        transformed = []
        last_partner = {}
        for (u, v) in pair_gates:
            emitted = False
            if u in giants:
                p = last_partner.get(u)
                if p is not None and p != v:
                    transformed.append((p, v))
                last_partner[u] = v
                emitted = True
            if v in giants:
                p = last_partner.get(v)
                if p is not None and p != u:
                    transformed.append((p, u))
                last_partner[v] = u
                emitted = True
            if not emitted:
                transformed.append((u, v))
        work_gates = transformed
    else:
        work_gates = pair_gates

    # interaction graph of the working (possibly transformed) sequence
    W2 = defaultdict(float)
    nbrs2 = defaultdict(set)
    first_use2 = {}
    touches = defaultdict(int)
    for r, (u, v) in enumerate(work_gates):
        W2[wkey(u, v)] += 1.0 / (1 + touches[u] + touches[v])
        touches[u] += 1
        touches[v] += 1
        nbrs2[u].add(v)
        nbrs2[v].add(u)
        if u not in first_use2:
            first_use2[u] = r
        if v not in first_use2:
            first_use2[v] = r
    count2 = defaultdict(float)
    for (u, v) in work_gates:
        count2[wkey(u, v)] += 1.0
    active2 = sorted(nbrs2.keys())

    pos = {}
    occupied = set()

    # ---- Stage 3: exact embedding of the transformed graph -----------------
    if giants and active2:
        # If the transformed graph is a disjoint union of simple paths,
        # lay it directly along the device's long wall-following path:
        # that embedding is not just valid but *straight*, which keeps
        # the giant's walk monotone (a wiggly exact embedding is valid
        # too, but the walking hub would zig-zag).
        if all(len(nbrs2[q]) <= 2 for q in active2):
            cyc = False
            seen_c = set()
            chains = []
            for q in active2:
                if q in seen_c or len(nbrs2[q]) == 2:
                    continue
                chain = [q]
                seen_c.add(q)
                cur, prev = q, None
                while True:
                    nxt = [x for x in nbrs2[cur] if x != prev]
                    if not nxt:
                        break
                    prev, cur = cur, nxt[0]
                    if cur in seen_c:
                        cyc = True
                        break
                    chain.append(cur)
                    seen_c.add(cur)
                chains.append(chain)
            if not cyc and len(seen_c) == len(active2):
                for c in chains:
                    # walk in temporal direction
                    if (first_use2.get(c[-1], 0)
                            < first_use2.get(c[0], 0)):
                        c.reverse()
                chains.sort(key=lambda c: min(first_use2.get(x, 0)
                                              for x in c))
                snake = _long_hardware_path(backend, phys_size)
                flat = [x for c in chains for x in c]
                if len(flat) <= len(snake):
                    off = _best_path_offset(flat, snake, W2,
                                            distance_matrix)
                    for i, q in enumerate(flat):
                        pos[q] = snake[off + i]
                    occupied = set(pos.values())
        if not pos:
            emb2 = try_embed(nbrs2, count2, active2)
            if emb2 is not None:
                pos = dict(emb2)
                occupied = set(pos.values())
        if not pos:
            # Trail graph does not embed (e.g. a giant's interleaved
            # visit pattern forms a triangular ladder). Minimise its
            # bandwidth with a Cuthill-McKee order and lay it along the
            # device path: low-bandwidth edges land on nearby slots of
            # the comb, which is exactly the 2-rail corridor the
            # interleaved walk wants.
            visited = set()
            rcm = []
            roots = sorted(active2, key=lambda q: (first_use2.get(q, 0), q))
            for root in roots:
                if root in visited:
                    continue
                dq = deque([root])
                visited.add(root)
                while dq:
                    x = dq.popleft()
                    rcm.append(x)
                    for y in sorted(nbrs2[x],
                                    key=lambda z: (len(nbrs2[z]),
                                                   first_use2.get(z, 0), z)):
                        if y not in visited:
                            visited.add(y)
                            dq.append(y)
            rank = {q: i for i, q in enumerate(rcm)}
            bw = max((abs(rank[u] - rank[v])
                      for u in active2 for v in nbrs2[u]), default=0)
            # Overall pair transience of the original circuit: if every
            # pair is one-shot (sigma ~ 0) the corridor still beats a
            # cluster even at higher bandwidth, because nothing recurs;
            # with recurring pairs the cluster (Stage 4) wins.
            num = den = 0.0
            for s in spans.values():
                num += s[2] * ((s[1] - s[0]) / max(M2 - 1, 1))
                den += s[2]
            sigma_all = num / den if den else 0.0
            # The wall-following path is a width-2 comb: slots up to 3
            # ranks apart stay within hardware distance ~2. A trail
            # graph with larger bandwidth gains from the corridor only
            # when the whole circuit is transient.
            snake = _long_hardware_path(backend, phys_size)
            if bw <= 3 and len(rcm) <= len(snake):
                off = _best_path_offset(rcm, snake, W2, distance_matrix,
                                        tie='centre')
                for i, q in enumerate(rcm):
                    pos[q] = snake[off + i]
                occupied = set(pos.values())
            elif sigma_all < 0.1:
                # Transient circuit whose trail came out high-bandwidth:
                # the trail abstraction failed, so band-order the
                # *original* interaction graph (hubs included as
                # ordinary band members) along the comb instead.
                W0 = defaultdict(float)
                t0 = defaultdict(int)
                for (u, v) in pair_gates:
                    W0[wkey(u, v)] += 1.0 / (1 + t0[u] + t0[v])
                    t0[u] += 1
                    t0[v] += 1
                fu0 = {}
                for r, (u, v) in enumerate(pair_gates):
                    for x in (u, v):
                        if x not in fu0:
                            fu0[x] = r
                visited = set()
                rcm0 = []
                for root in sorted(active,
                                   key=lambda q: (len(neighbors[q]),
                                                  fu0.get(q, 0), q)):
                    if root in visited:
                        continue
                    dq = deque([root])
                    visited.add(root)
                    while dq:
                        x = dq.popleft()
                        rcm0.append(x)
                        for y in sorted(neighbors[x],
                                        key=lambda z: (len(neighbors[z]),
                                                       fu0.get(z, 0), z)):
                            if y not in visited:
                                visited.add(y)
                                dq.append(y)
                if len(rcm0) <= len(snake):
                    off = _best_path_offset(rcm0, snake, W0,
                                            distance_matrix, tie='centre')
                    for i, q in enumerate(rcm0):
                        pos[q] = snake[off + i]
                    occupied = set(pos.values())

    # ---- Stage 4: cluster placement + influence-decayed descent ------------
    if not pos and active2:
        hw_root = _pick_hardware_root(backend, distance_matrix, phys_size)
        w_deg = {q: sum(W2[wkey(q, v)] for v in nbrs2[q]) for q in active2}

        region_sites = None     # set later only if the free cluster is a tree

        def build_cluster(start_q):
            """One greedy max-attachment construction (seeded at
            ``start_q``) followed by the exchange descent. Returns
            (pos, occupied, static_cost)."""
            cpos = {}
            cocc = set()

            def place_cost(q, p):
                c = 0.0
                for v in nbrs2[q]:
                    if v in cpos:
                        c += W2[wkey(q, v)] * dist(p, cpos[v])
                return c

            attach = defaultdict(float)
            unplaced = set(active2)
            while unplaced:
                if not cpos and start_q is not None:
                    q = start_q
                else:
                    cand = [x for x in unplaced if attach[x] > 0]
                    if cand:
                        q = min(cand, key=lambda x: (-attach[x],
                                                     first_use2.get(x, 0), x))
                    else:
                        q = min(unplaced,
                                key=lambda x: (-w_deg[x],
                                               first_use2.get(x, 0), x))
                unplaced.discard(q)

                allowed = (set(region_sites) if region_sites is not None
                           else None)
                if nbrs2[q] & cpos.keys():
                    placed_nb_sites = [cpos[v] for v in nbrs2[q]
                                       if v in cpos]
                    cset = set()
                    for s in placed_nb_sites:
                        for a in backend[s]:
                            if a < phys_size and a not in cocc:
                                cset.add(a)
                            for b in backend[a]:
                                if b < phys_size and b not in cocc:
                                    cset.add(b)
                    if allowed is not None:
                        cset &= allowed
                        if not cset:
                            cset = {p for p in allowed if p not in cocc}
                    if not cset:
                        cset = {p for p in range(phys_size)
                                if p not in cocc}
                    # tie-break toward sites with more occupied
                    # neighbours: closes cycles in the occupied region,
                    # which the router needs to rotate tokens without
                    # stranding them (decisive on sparse devices).
                    best = min(cset, key=lambda p: (
                        place_cost(q, p),
                        -sum(1 for a in backend[p] if a in cocc), p))
                else:
                    free = [p for p in range(phys_size) if p not in cocc]
                    if allowed is not None:
                        rfree = [p for p in free if p in allowed]
                        if rfree:
                            free = rfree
                    if not free:
                        break

                    def anchor_key(p):
                        fd = sum(1 for a in backend[p] if a not in cocc)
                        return (-fd, dist(p, hw_root), p)
                    best = min(free, key=anchor_key)
                cpos[q] = best
                cocc.add(best)
                for v in nbrs2[q]:
                    if v in unplaced:
                        attach[v] += W2[wkey(q, v)]

            # deterministic exchange descent on sum W2 * (d - 1)
            def vertex_cost(q, p, skip=None):
                c = 0.0
                for v in nbrs2[q]:
                    if v == skip or v not in cpos:
                        continue
                    c += W2[wkey(q, v)] * (dist(p, cpos[v]) - 1)
                return c

            qubits = sorted(cpos.keys())
            for _ in range(len(qubits) + 1):
                improved = False
                if region_sites is not None:
                    free_sites = [p for p in region_sites
                                  if p not in cocc]
                else:
                    free_sites = [p for p in range(phys_size)
                                  if p not in cocc]
                for q in qubits:
                    cur = vertex_cost(q, cpos[q])
                    if cur == 0:
                        continue
                    best_p, best_c = None, cur
                    for p in free_sites:
                        c = vertex_cost(q, p)
                        if c < best_c - 1e-12:
                            best_c, best_p = c, p
                    if best_p is not None:
                        cocc.discard(cpos[q])
                        cocc.add(best_p)
                        free_sites.remove(best_p)
                        free_sites.append(cpos[q])
                        cpos[q] = best_p
                        improved = True
                # exchanges restricted to nearby seats: a swap between
                # two distant seats is two relocations in disguise, and
                # the relocation pass above already covers those.
                for i in range(len(qubits)):
                    qi = qubits[i]
                    for j in range(i + 1, len(qubits)):
                        qj = qubits[j]
                        pi, pj = cpos[qi], cpos[qj]
                        if dist(pi, pj) > 2:
                            continue
                        old = (vertex_cost(qi, pi, skip=qj)
                               + vertex_cost(qj, pj, skip=qi))
                        new = (vertex_cost(qi, pj, skip=qj)
                               + vertex_cost(qj, pi, skip=qi))
                        if new < old - 1e-12:
                            cpos[qi], cpos[qj] = pj, pi
                            improved = True
                if not improved:
                    break

            cost = sum(W2[k] * (dist(cpos[k[0]], cpos[k[1]]) - 1)
                       for k in W2 if k[0] in cpos and k[1] in cpos)
            return cpos, cocc, cost

        # Multi-start: the greedy construction is sensitive to its seed
        # qubit, and on small registers the static objective ranks the
        # resulting *sibling* layouts reliably (they share one geometry
        # family), so every active qubit is tried. On large registers
        # the routed cost is congestion-dominated and the static
        # ranking is no longer trustworthy (verified on the 75-qubit
        # multiplier, where it picks a 12%-worse sibling), so a single
        # deterministic start is kept.
        def _cyclomatic(occ):
            oset = set(occ)
            E = sum(1 for a in occ for b in backend[a] if b in oset) // 2
            seen = set()
            comp = 0
            for x in occ:
                if x in seen:
                    continue
                comp += 1
                stack = [x]
                while stack:
                    y = stack.pop()
                    if y in seen:
                        continue
                    seen.add(y)
                    stack.extend(z for z in backend[y] if z in oset)
            return E - len(oset) + comp

        def run_multistart():
            if len(active2) <= 30:
                br = None
                for s in active2:
                    cpos, cocc, cost = build_cluster(s)
                    if br is None or cost < br[2]:
                        br = (cpos, cocc, cost)
                return br
            dr = build_cluster(None)
            br = (dr[0], dr[1], dr[2])
            peaks = [q for q in active2
                     if all(w_deg[q] >= w_deg[v] for v in nbrs2[q])]
            for s in peaks:
                cpos, cocc, cost = build_cluster(s)
                if cost < 0.95 * br[2]:
                    br = (cpos, cocc, cost)
            return br

        # Note: a face-anchored region rebuild for tree-shaped occupied
        # regions was evaluated (see report) — it traded wins and
        # losses with no reliable static arbiter, so only the
        # cycle-closing placement tie-break is kept.
        best_run = run_multistart()
        pos, occupied = best_run[0], best_run[1]

    # ---- seat the giants ----------------------------------------------------
    # Each giant sits next to its first partner: that is where its walk
    # begins; the trail layout takes care of the rest of the journey.
    for h in sorted(giants,
                    key=lambda q: (min((r for r, g in enumerate(pair_gates)
                                        if q in g), default=0), q)):
        if h in pos:
            continue
        free = [p for p in range(phys_size) if p not in occupied]
        if not free:
            break
        first_partner = None
        for (u, v) in pair_gates:
            if u == h and v in pos:
                first_partner = v
                break
            if v == h and u in pos:
                first_partner = u
                break
        if first_partner is not None:
            anchor = pos[first_partner]
            best = min(free, key=lambda p: (dist(p, anchor), p))
        else:
            best = min(free, key=lambda p: (
                -sum(1 for a in backend[p] if a not in occupied), p))
        pos[h] = best
        occupied.add(best)

    # any remaining active qubits (e.g. isolated after transformation)
    for q in active:
        if q in pos:
            continue
        free = [p for p in range(phys_size) if p not in occupied]
        if not free:
            break
        best = min(free)
        pos[q] = best
        occupied.add(best)

    return _finalize(pos)


def generate_vex_initial_mapping(qasm_code, backend_edges, num_qubits,
                                 access, distance_matrix, backend):
    """VEX — initial mapping by Virtual EXecution.

    A static pairwise-distance objective is blind to drift: the router
    moves qubits while the circuit runs, so a pair that interacts late
    does not need to start close — it needs to be close to where its
    partner *will be*. VEX therefore builds the initial mapping by
    virtually executing the 2-qubit gate sequence once:

    * Each logical qubit is placed permanently (= its initial-mapping
      site) at the moment of its first 2q gate, choosing the free site
      that minimises the weighted distance to the *current virtual
      positions* of all its already-placed interaction partners
      (weights = total interaction counts, a static lookahead).
    * After every gate, the two tokens walk toward each other along a
      deterministic shortest path (true token swaps), so subsequent
      placement decisions see the drifted state the real router will
      produce, not the time-0 snapshot.

    Stage A still tries an exact subgraph embedding first: if the whole
    interaction graph fits the hardware, that mapping needs zero SWAPs.

    Single pass over the gate list, deterministic, parameter-free.
    """
    weights, neighbors = _interaction_graph(access)
    if not weights:
        return generate_trivial_initial_mapping(num_qubits)

    active = sorted(neighbors.keys())
    phys_size = len(distance_matrix)

    def _finalize(log_to_phys):
        new_map = [-1] * num_qubits
        new_rev = [-1] * num_qubits
        used = set()
        for lq, pq in log_to_phys.items():
            if 0 <= lq < num_qubits and 0 <= pq < phys_size:
                new_map[lq] = pq
                new_rev[pq] = lq
                used.add(pq)
        for lq in range(num_qubits):
            if new_map[lq] != -1:
                continue
            for p in range(phys_size):
                if p not in used:
                    new_map[lq] = p
                    new_rev[p] = lq
                    used.add(p)
                    break
        return new_map, new_rev

    # ---- Stage A: exact zero-SWAP embedding --------------------------------
    embed = _subgraph_embed(neighbors, weights, active, backend,
                            phys_size, 200000)
    if embed is not None:
        return _finalize(embed)

    # ---- Stage B: virtual execution ----------------------------------------
    def wkey(u, v):
        return (u, v) if u < v else (v, u)

    BIG = phys_size + 1

    def dist(p, q):
        d = distance_matrix[p][q]
        return BIG if d == float('inf') else d

    hw_root = _pick_hardware_root(backend, distance_matrix, phys_size)

    gates = []
    for key in sorted(access.keys()):
        qs = access[key]
        if len(qs) == 2 and qs[0] != qs[1]:
            gates.append(tuple(qs))

    init_site = {}          # logical -> permanent initial site
    vpos = {}               # logical -> current virtual site
    site_token = {}         # site -> logical currently there

    def free_sites():
        return [p for p in range(phys_size)
                if p not in site_token and p not in init_site.values()]

    def free_deg(p):
        return sum(1 for a in backend[p] if a not in site_token)

    def place(q, anchor=None):
        """Choose q's permanent site. anchor = preferred adjacency site."""
        placed_partners = [(v, vpos[v]) for v in neighbors[q] if v in vpos]
        candidates = [p for p in range(phys_size)
                      if p not in site_token
                      and p not in init_used]
        if not candidates:
            return False
        if placed_partners:
            def key(p):
                c = 0.0
                for v, pv in placed_partners:
                    c += weights[wkey(q, v)] * dist(p, pv)
                return (c, -free_deg(p), p)
        elif anchor is not None:
            def key(p):
                return (dist(p, anchor), -free_deg(p), p)
        else:
            def key(p):
                return (dist(p, hw_root), -free_deg(p), p)
        s = min(candidates, key=key)
        init_site[q] = s
        init_used.add(s)
        vpos[q] = s
        site_token[s] = q
        return True

    def walk_together(u, v):
        """Token-swap u and v toward each other along a deterministic
        shortest path until adjacent."""
        while True:
            pu, pv = vpos[u], vpos[v]
            d = dist(pu, pv)
            if d <= 1:
                return
            # step u one hop toward v (smallest-id improving neighbour)
            nxt = min((n for n in backend[pu] if dist(n, pv) < d),
                      default=None)
            if nxt is None:
                return
            other = site_token.get(nxt)
            del site_token[pu]
            if other is not None:
                site_token[pu] = other
                vpos[other] = pu
            site_token[nxt] = u
            vpos[u] = nxt
            u, v = v, u  # alternate ends so they meet in the middle

    init_used = set()
    last_focus = None
    for (u, v) in gates:
        up, vp = u in vpos, v in vpos
        if not up and not vp:
            # heavier qubit first, anchored near the last activity
            wu = sum(weights[wkey(u, x)] for x in neighbors[u])
            wv = sum(weights[wkey(v, x)] for x in neighbors[v])
            first, second = (u, v) if (wu, -u) >= (wv, -v) else (v, u)
            place(first, anchor=last_focus)
            place(second, anchor=vpos[first])
        elif not up:
            place(u, anchor=vpos[v])
        elif not vp:
            place(v, anchor=vpos[u])
        if u in vpos and v in vpos:
            walk_together(u, v)
            last_focus = vpos[u]

    return _finalize(init_site)


def generate_cblc_initial_mapping(qasm_code, backend_edges, num_qubits,
                                  access, distance_matrix, backend):
    
    weights, neighbors = _interaction_graph(access)
    if not weights:
        # No 2q interactions: trivial mapping is optimal.
        return generate_trivial_initial_mapping(num_qubits)

    active = sorted(neighbors.keys())
    phys_size = len(distance_matrix)

    # ---- Structural dispatch (hardware-derived, not tuned) ------------------
    # Hardware diameter — the longest shortest-path on the device.
    def _hw_diameter():
        m = 0
        for row in distance_matrix:
            for d in row:
                if d != float('inf') and d > m:
                    m = d
        return m
    hw_diam = _hw_diameter()

    # Logical-interaction-graph diameter — longest shortest-path over
    # the 2q-interaction graph (computed only among active qubits).
    def _log_diameter():
        m = 0
        for start in active:
            d = {start: 0}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in neighbors[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
                        if d[v] > m:
                            m = d[v]
        return m
    log_diam = _log_diameter()

    def _finalize(log_to_phys):
        """Build (mapping, reverse) from a partial logical->physical
        dict, then deterministically fill idle/unplaced qubits."""
        new_map = [-1] * num_qubits
        new_rev = [-1] * num_qubits
        used = set()
        for lq, pq in log_to_phys.items():
            if 0 <= lq < num_qubits and 0 <= pq < phys_size:
                new_map[lq] = pq
                new_rev[pq] = lq
                used.add(pq)
        for lq in range(num_qubits):
            if new_map[lq] != -1:
                continue
            for p in range(phys_size):
                if p not in used:
                    new_map[lq] = p
                    new_rev[p] = lq
                    used.add(p)
                    break
        return new_map, new_rev

    # ---- Stage 1: exact zero-SWAP subgraph embedding ------------------------
    # The strongest possible initial mapping is one under which *every*
    # two-qubit gate already lands on a coupled physical pair — i.e. the
    # logical interaction graph is a subgraph of the hardware. When such
    # an embedding exists the routed SWAP count is zero. A bounded,
    # deterministic VF2-style search recovers it; this is exactly where
    # a randomised routing-driven layout (Sabre) leaves SWAPs on the
    # table on otherwise perfectly-embeddable circuits. The search
    # fails fast (degree pruning) on hub/dense graphs, so the budget
    # is rarely consumed.
    step_budget = 40000
    embed = _subgraph_embed(neighbors, weights, active, backend,
                            phys_size, step_budget)
    if embed is not None:
        return _finalize(embed)

    # ---- Stage 2: hardware-derived structural dispatch ----------------------
    if log_diam <= hw_diam:
        # Mesh-like dense graphs: denser than the device on average yet
        # with genuine 2D extent (diameter >= 3, so not a collapsed
        # star or near-clique). For these a global spectral co-embedding
        # aligns the logical interaction geometry to the device's own
        # geometry far better than a local routing-driven layout. Both
        # tests are structural / hardware-derived (mean degrees and a
        # fixed minimum extent), not tuned magnitudes.
        avg_hw_deg = (sum(len(backend[p]) for p in backend)
                      / max(len(backend), 1))
        avg_log_deg = (sum(len(neighbors[q]) for q in active)
                       / max(len(active), 1))
        if avg_log_deg > avg_hw_deg and log_diam >= 3:
            return _spectral_coembed(
                weights, neighbors, active, backend, phys_size,
                num_qubits, ndim=2)
        # Otherwise the interaction graph fits within a hardware
        # neighbourhood (star / near-clique / sparse-compact) and
        # Sabre's routing-driven search has full room to find a
        # low-cost layout; defer to it.
        return generate_sabre_initial_mapping(
            qasm_code, backend_edges, num_qubits)

    # CBLC only earns its keep when the logical graph is *stretched*
    # relative to the device (diameter exceeds the hardware diameter),
    # where a routing-driven layout struggles to keep a long
    # interaction chain locally contiguous.

    # ---- Concentric BFS-Layer Coupling --------------------------------------
    # Logical weighted degree drives both the layer-0 root and the
    # forward-load ordering inside each layer.
    w_deg = {q: sum(weights[(q, v) if q < v else (v, q)] for v in neighbors[q])
             for q in active}

    # Hardware root + BFS distances from it (the "concentric" anchor).
    hw_root = _pick_hardware_root(backend, distance_matrix, phys_size)
    hw_dist_from_root = _hardware_bfs_from(backend, hw_root, phys_size)
    # Default unreachable distance to a large but finite value to keep
    # arithmetic well defined.
    BIG = phys_size + 1

    # Residual hardware degree: how many of each physical node's
    # neighbours are still free. Consumed by every placement.
    residual_deg = {p: len(backend[p]) for p in backend}

    new_map = [-1] * num_qubits
    new_rev = [-1] * num_qubits
    used_phys = set()
    placed = set()

    def place(log_q, phys_q):
        new_map[log_q] = phys_q
        new_rev[phys_q] = log_q
        used_phys.add(phys_q)
        placed.add(log_q)
        # Consume one unit of free degree on each hardware neighbour
        # of phys_q. This is the residual-degree budgeting step.
        for nb in backend[phys_q]:
            if nb in residual_deg:
                residual_deg[nb] -= 1

    # Process each connected component of the logical interaction graph
    # independently, in descending order of total weight (largest +
    # heaviest component first — it deserves the central region).
    seen = set()
    components = []
    for q in active:
        if q in seen:
            continue
        comp = []
        stack = [q]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            for v in neighbors[x]:
                if v not in seen and v in neighbors:
                    stack.append(v)
        components.append(comp)

    def comp_weight(c):
        s = 0.0
        cset = set(c)
        for u in c:
            for v in neighbors[u]:
                if v in cset and u < v:
                    s += weights[(u, v)]
        return s

    components.sort(key=lambda c: (-comp_weight(c), -len(c), min(c)))

    # First component is anchored at the hardware root; later components
    # are anchored at the most central still-free physical qubit.
    first_component = True

    def score_phys_for(log_q, target_depth):
        """Deterministic lex-key for placing log_q on each free physical p.

        Smaller key = better. The key is a tuple:

            (weighted_distance_to_placed_neighbours,
             |hw_depth - target_depth|,
             best-fit residual-degree mismatch,
             physical id)

        * **First component**: the dominant cost. It is the *exact*
          future SWAP cost contributed by edges from ``log_q`` to its
          already-placed neighbours. Edges with weight ``w_e`` at
          distance ``d`` contribute ``w_e * d``.
        * **Second component**: concentric layer alignment. Only acts
          when multiple physical positions tie on the primary cost,
          breaking the tie in favour of the position whose hardware
          depth matches the logical BFS depth.
        * **Third component**: hardware free-degree budgeting. This is
          the novel routing-resource term: we prefer a free physical
          whose residual free degree best-fits ``log_q``'s remaining
          unplaced-neighbour count. Hardware degree is treated as a
          routing capacity consumed by placements; mismatched bins are
          wasted capacity.
        * **Fourth component**: deterministic id tiebreak.
        """
        placed_nbrs = [nb for nb in neighbors[log_q] if nb in placed]
        # Future demand for adjacency that this placement must support.
        unplaced_nbr_count = sum(
            1 for nb in neighbors[log_q] if nb not in placed)

        best_p = None
        best_key = None
        for p in range(phys_size):
            if p in used_phys:
                continue
            d_cost = 0
            for nb in placed_nbrs:
                w_e = weights[(log_q, nb) if log_q < nb else (nb, log_q)]
                d = distance_matrix[p][new_map[nb]]
                if d == float('inf'):
                    d = BIG
                d_cost += w_e * d
            depth_p = hw_dist_from_root.get(p, BIG)
            layer_gap = abs(depth_p - target_depth)
            # Best-fit residual mismatch: how far p's free degree is
            # from log_q's remaining unplaced-neighbour count. We want
            # residual_deg(p) >= unplaced_nbr_count when possible;
            # if not, we still prefer the closest match.
            r = residual_deg.get(p, 0)
            if r >= unplaced_nbr_count:
                resid_mismatch = r - unplaced_nbr_count
            else:
                # Heavy penalty for under-fit; surplus is cheap.
                resid_mismatch = (unplaced_nbr_count - r) * 4
            key = (d_cost, layer_gap, resid_mismatch, p)
            if best_key is None or key < best_key:
                best_key = key
                best_p = p
        return best_p

    for comp in components:
        # Root of this component = highest weighted-degree node.
        root_log = max(comp, key=lambda q: (w_deg[q], -q))
        if first_component:
            root_phys = hw_root if hw_root not in used_phys and hw_root < phys_size else None
            if root_phys is None:
                # Fall back to any still-free, low-eccentricity physical.
                root_phys = score_phys_for(root_log, 0)
            first_component = False
        else:
            # Anchor at the still-free physical position closest to the
            # original hardware root (preserves the "concentric"
            # interpretation across components).
            root_phys = None
            best_d = None
            for p in range(phys_size):
                if p in used_phys:
                    continue
                d = hw_dist_from_root.get(p, BIG)
                key = (d, -residual_deg.get(p, 0), p)
                if best_d is None or key < best_d:
                    best_d = key
                    root_phys = p
        if root_phys is None:
            continue
        place(root_log, root_phys)

        # Compute BFS layers of THIS component, rooted at root_log.
        comp_neighbors = {q: {v for v in neighbors[q] if v in comp} for q in comp}
        layer_of_local, layers_local = _logical_bfs_layers(
            comp_neighbors, weights, set(comp), root_log)

        # Layer-by-layer placement.
        for depth, layer in enumerate(layers_local):
            if depth == 0:
                # root_log already placed.
                continue
            # Forward-load order within the layer: sum of weights to all
            # neighbours NOT YET placed at this point in time. This is
            # the deterministic ordering principle.
            def fwd_load(q):
                load = 0.0
                for nb in comp_neighbors[q]:
                    if nb not in placed:
                        load += weights[(q, nb) if q < nb else (nb, q)]
                return load
            # Stable sort: heaviest forward load first; ties → id.
            ordered_layer = sorted(layer, key=lambda q: (-fwd_load(q), q))
            for log_q in ordered_layer:
                if log_q in placed:
                    continue
                target_depth = layer_of_local[log_q]
                best_p = score_phys_for(log_q, target_depth)
                if best_p is None:
                    break
                place(log_q, best_p)

        # Any logical in this component left unplaced (e.g. depth
        # explosion fell out of the BFS due to fragmentation) gets the
        # best remaining slot.
        for log_q in comp:
            if log_q in placed:
                continue
            best_p = score_phys_for(log_q, 0)
            if best_p is None:
                break
            place(log_q, best_p)

    # Idle logical qubits (only ever appear in 1-qubit gates) take the
    # next available free physical positions deterministically.
    for log_q in range(num_qubits):
        if new_map[log_q] != -1:
            continue
        for p in range(phys_size):
            if p not in used_phys:
                new_map[log_q] = p
                new_rev[p] = log_q
                used_phys.add(p)
                break

    return new_map, new_rev


def swap_logical_physical_mappings(logical_to_physical, physical_to_logical, swap_pair, inplace=False):
    """Swap two physical qubits in the mapping pair.

    With ``inplace=False`` a copy of ``logical_to_physical`` is returned
    and ``physical_to_logical`` is left untouched; with ``inplace=True``
    both lists are updated in place.
    """
    updated_mapping = logical_to_physical if inplace else logical_to_physical[:]
    physical_1, physical_2 = swap_pair

    logical_1 = physical_to_logical[physical_1]
    logical_2 = physical_to_logical[physical_2]

    if logical_1 != -1:
        updated_mapping[logical_1] = physical_2
    if logical_2 != -1:
        updated_mapping[logical_2] = physical_1

    if inplace:
        physical_to_logical[physical_1] = logical_2
        physical_to_logical[physical_2] = logical_1

    return updated_mapping


def swap_logical_physical_isl_mapping(isl_mapping, swap_pair):
    """Swap two physical qubits in an ISL Map mapping (used in use_isl mode)."""
    import islpy as isl
    q1, q2 = swap_pair
    swap_domain = isl.Set(f"{{[{q1}];[{q2}]}}")
    swap_map = isl.Map(f"{{[{q1}] -> [{q2}]; [{q2}] -> [{q1}]}}")
    other_mapping = isl_mapping.subtract_range(swap_domain)
    return isl_mapping.apply_range(swap_map).union(other_mapping)


def swap_logical_physical_isl_mapping_path(isl_mapping, swap_path_map):
    """Apply a multi-step ISL swap-path map to an ISL logical-to-physical map."""
    if swap_path_map.is_empty():
        return isl_mapping
    other_mapping = isl_mapping.subtract_range(swap_path_map.domain())
    return isl_mapping.apply_range(swap_path_map).union(other_mapping)


def _device_faces(backend, phys_size):
    """Chordless minimum-cycle-basis faces of the hardware graph,
    each returned in cyclic order."""
    import networkx as nx
    G = nx.Graph()
    for a in backend:
        for b in backend[a]:
            if a < b and a < phys_size and b < phys_size:
                G.add_edge(a, b)
    faces = nx.minimum_cycle_basis(G)

    def order_cycle(g, nodes):
        sub = g.subgraph(nodes)
        if any(d != 2 for _, d in sub.degree()):
            return None
        start = sorted(nodes)[0]
        order = [start]
        prev = None
        cur = start
        while len(order) < len(nodes):
            nxts = [x for x in sub.neighbors(cur) if x != prev]
            if not nxts:
                return None
            prev, cur = cur, nxts[0]
            order.append(cur)
        return order
    out = []
    for f in faces:
        oc = order_cycle(G, f)
        if oc:
            out.append(oc)
    return out, order_cycle


def generate_carousel_candidates(access, distance_matrix, backend,
                                 num_qubits):
    """Lane-carousel candidates for ring-structured persistent circuits.

    If the logical interaction graph decomposes into two chordless
    rings (plus a few cross edges) and the device has faces long
    enough to host them, place each ring along the arc of one of two
    adjacent faces, leaving the shared bridge free as a swap lane.
    Configurations are scored by weighted stretch with a temporal
    tie-break; the best two (from distinct face pairs) are returned as
    complete mappings. Returns [] when the structure is absent.
    """
    import networkx as nx
    phys_size = len(distance_matrix)

    def wkey(u, v):
        return (u, v) if u < v else (v, u)

    pair_gates = []
    for key in sorted(access.keys()):
        qs = access[key]
        if len(qs) == 2 and qs[0] != qs[1]:
            pair_gates.append(tuple(qs))
    C = defaultdict(float)
    first = {}
    nbrs = defaultdict(set)
    for r, (u, v) in enumerate(pair_gates):
        k = wkey(u, v)
        C[k] += 1
        if k not in first:
            first[k] = r
        nbrs[u].add(v)
        nbrs[v].add(u)
    if not C or any(len(nbrs[q]) > 3 for q in nbrs):
        return []
    M = len(pair_gates)
    IG = nx.Graph(list(C.keys()))
    rings = [c for c in nx.minimum_cycle_basis(IG) if 6 <= len(c) <= 12]
    if len(rings) < 2:
        return []

    faces, order_cycle = _device_faces(backend, phys_size)
    faces = [f for f in faces
             if len(f) >= max(len(rings[0]), len(rings[1])) + 3]
    if not faces:
        return []
    fsets = [set(f) for f in faces]
    fpairs = [(i, j) for i in range(len(faces)) for j in range(len(faces))
              if i < j and len(fsets[i] & fsets[j]) == 3]
    if not fpairs:
        return []

    G2 = nx.Graph(list(C.keys()))
    R1 = order_cycle(G2, rings[0])
    R2 = order_cycle(G2, rings[1])
    if not R1 or not R2 or set(R1) & set(R2):
        return []

    def ring_configs(R):
        out = []
        n = len(R)
        for rot in range(n):
            for flip in (False, True):
                r = R[rot:] + R[:rot]
                if flip:
                    r = r[::-1]
                out.append(r)
        return out

    def arc_of(F, shared):
        L = len(F)
        for k in range(L):
            if F[k] in shared and F[(k + 1) % L] not in shared:
                start = (k + 1) % L
                return [F[(start + t) % L] for t in range(L)
                        if F[(start + t) % L] not in shared]
        return None

    n1, n2 = len(R1), len(R2)
    per_pair = {}
    per_cut = {}
    for i, j in fpairs:
      shared = fsets[i] & fsets[j]
      for (Ra, Rb) in ((R1, R2), (R2, R1)):
        RC1 = ring_configs(Ra)
        RC2 = ring_configs(Rb)
        na, nb = len(Ra), len(Rb)
        A1 = arc_of(faces[i], shared)
        A2 = arc_of(faces[j], shared)
        if not A1 or not A2 or len(A1) < na or len(A2) < nb:
            continue
        a1opts = (A1[:na], A1[len(A1) - na:])
        a2opts = (A2[:nb], A2[len(A2) - nb:])
        for a1 in a1opts:
            P1 = [dict(zip(rc, a1)) for rc in RC1]
            for a2 in a2opts:
                for rc2 in RC2:
                    p2 = dict(zip(rc2, a2))
                    for p1 in P1:
                        pos = {**p1, **p2}
                        t = tp = 0.0
                        ok = True
                        for k, w in C.items():
                            if k[0] not in pos or k[1] not in pos:
                                ok = False
                                break
                            d = distance_matrix[pos[k[0]]][pos[k[1]]]
                            t += w * (d - 1)
                            if d == 1:
                                tp += first[k] / M
                        if not ok:
                            continue
                        key = (t, tp)
                        # the cut edge (the ring edge spanning the lane)
                        # determines the steady-state dynamics; keep the
                        # best config per (face pair, cut) so the
                        # portfolio offers dynamically distinct options
                        rc1 = list(p1.keys())
                        cut = (i, j, min(rc1[0], rc1[-1]),
                               max(rc1[0], rc1[-1]))
                        cur = per_cut.get(cut)
                        if cur is None or key < cur[1]:
                            per_cut[cut] = (pos, key)
                        cur = per_pair.get((i, j))
                        if cur is None or key < cur[1]:
                            per_pair[(i, j)] = (pos, key)
    if not per_pair:
        return []
    ranked = sorted(per_pair.values(), key=lambda x: x[1])

    def finalize(log_to_phys):
        new_map = [-1] * num_qubits
        used = set()
        for lq, pq in log_to_phys.items():
            if 0 <= lq < num_qubits and 0 <= pq < phys_size:
                new_map[lq] = pq
                used.add(pq)
        for lq in range(num_qubits):
            if new_map[lq] != -1:
                continue
            for p in range(phys_size):
                if p not in used:
                    new_map[lq] = p
                    used.add(p)
                    break
        return new_map

    best_pair = min(per_pair.items(), key=lambda kv: kv[1][1])[0]
    # top-4 dynamically distinct cuts of the best face pair
    cuts = sorted((v for k, v in per_cut.items()
                   if (k[0], k[1]) == best_pair), key=lambda x: x[1])[:4]
    out = [finalize(v[0]) for v in cuts]
    # plus the best config of the runner-up face pair
    others = sorted((v for k, v in per_pair.items() if k != best_pair),
                    key=lambda x: x[1])
    if others:
        out.append(finalize(others[0][0]))
    return out


def generate_cluster_variant_mapping(access, distance_matrix, backend,
                                     num_qubits, mode='raw',
                                     breathe=False, top2=False):
    """Stage-4-style cluster with selectable weighting (raw counts or
    influence decay) and optional breathing constraint (every occupied
    site keeps a free neighbour). Used as portfolio candidates for
    ring-structured persistent circuits."""
    phys_size = len(distance_matrix)

    def wkey(u, v):
        return (u, v) if u < v else (v, u)

    pair_gates = []
    for key in sorted(access.keys()):
        qs = access[key]
        if len(qs) == 2 and qs[0] != qs[1]:
            pair_gates.append(tuple(qs))
    W = defaultdict(float)
    nbrs = defaultdict(set)
    fu = {}
    t = defaultdict(int)
    for r, (u, v) in enumerate(pair_gates):
        base = 1.0 / (1 + t[u] + t[v])
        W[wkey(u, v)] += base if mode == 'decay' else 1.0
        t[u] += 1
        t[v] += 1
        nbrs[u].add(v)
        nbrs[v].add(u)
        for x in (u, v):
            if x not in fu:
                fu[x] = r
    if not W:
        return None
    act = sorted(nbrs)
    w_deg = {q: sum(W[wkey(q, v)] for v in nbrs[q]) for q in act}
    hw_root = _pick_hardware_root(backend, distance_matrix, phys_size)

    def breathing_ok(p, occ):
        occ2 = occ | {p}

        def enclosed(x):
            return all(y in occ2 for y in backend[x])
        if enclosed(p):
            return False
        for y in backend[p]:
            if y in occ and enclosed(y):
                return False
        return True

    def build(s):
        pos = {}
        occ = set()

        def pc(q, p):
            return sum(W[wkey(q, v)] * distance_matrix[p][pos[v]]
                       for v in nbrs[q] if v in pos)
        attach = defaultdict(float)
        unpl = set(act)
        while unpl:
            if not pos:
                q = s
            else:
                cand = [x for x in unpl if attach[x] > 0]
                q = (min(cand, key=lambda x: (-attach[x], fu.get(x, 0), x))
                     if cand else
                     min(unpl, key=lambda x: (-w_deg[x], fu.get(x, 0), x)))
            unpl.discard(q)
            nb = [pos[v] for v in nbrs[q] if v in pos]
            if nb:
                cs = set()
                for ss in nb:
                    for a in backend[ss]:
                        if a < phys_size and a not in occ:
                            cs.add(a)
                        for b in backend[a]:
                            if b < phys_size and b not in occ:
                                cs.add(b)
                if breathe:
                    bs = {p for p in cs if breathing_ok(p, occ)}
                    if bs:
                        cs = bs
                if not cs:
                    cs = {p for p in range(phys_size) if p not in occ}
                best = min(cs, key=lambda p: (
                    pc(q, p),
                    -sum(1 for a in backend[p] if a in occ), p))
            else:
                fr = [p for p in range(phys_size) if p not in occ]
                best = min(fr, key=lambda p: (
                    -sum(1 for a in backend[p] if a not in occ),
                    distance_matrix[p][hw_root], p))
            pos[q] = best
            occ.add(best)
            for v in nbrs[q]:
                if v in unpl:
                    attach[v] += W[wkey(q, v)]

        def vc(q, p, skip=None):
            c = 0.0
            for v in nbrs[q]:
                if v == skip or v not in pos:
                    continue
                c += W[wkey(q, v)] * (distance_matrix[p][pos[v]] - 1)
            return c
        qs = sorted(pos)
        for _ in range(len(qs) + 1):
            imp = False
            fr = [p for p in range(phys_size) if p not in occ]
            for q in qs:
                cur = vc(q, pos[q])
                if cur == 0:
                    continue
                bp, bc = None, cur
                for p in fr:
                    if breathe and not breathing_ok(p, occ - {pos[q]}):
                        continue
                    c = vc(q, p)
                    if c < bc - 1e-12:
                        bc, bp = c, p
                if bp is not None:
                    occ.discard(pos[q])
                    occ.add(bp)
                    fr.remove(bp)
                    fr.append(pos[q])
                    pos[q] = bp
                    imp = True
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    qi, qj = qs[i], qs[j]
                    pi, pj = pos[qi], pos[qj]
                    if distance_matrix[pi][pj] > 2:
                        continue
                    if (vc(qi, pj, skip=qj) + vc(qj, pi, skip=qi)
                            < vc(qi, pi, skip=qj) + vc(qj, pj, skip=qi)
                            - 1e-12):
                        pos[qi], pos[qj] = pj, pi
                        imp = True
            if not imp:
                break
        cost = sum(W[k] * (distance_matrix[pos[k[0]]][pos[k[1]]] - 1)
                   for k in W)
        return pos, cost

    runs = []
    if len(act) <= 30:
        for s in act:
            runs.append(build(s))
    else:
        runs.append(build(act[0]))
    runs.sort(key=lambda r: r[1])

    def finalize_map(pos):
        new_map = [-1] * num_qubits
        used = set()
        for lq, pq in pos.items():
            if 0 <= lq < num_qubits:
                new_map[lq] = pq
                used.add(pq)
        for lq in range(num_qubits):
            if new_map[lq] != -1:
                continue
            for p in range(phys_size):
                if p not in used:
                    new_map[lq] = p
                    used.add(p)
                    break
        return new_map

    if top2:
        out = [finalize_map(runs[0][0])]
        for pos, _ in runs[1:]:
            m = finalize_map(pos)
            if m != out[0]:
                out.append(m)
                break
        return out
    return finalize_map(runs[0][0])


def generate_tree_embed_mapping(access, distance_matrix, backend,
                                num_qubits):
    """Maximum-weight spanning tree of the interaction graph embedded
    exactly (trees always have girth-compatible embeddings), then a
    bounded exchange descent on the full weighted stretch. Portfolio
    candidate for ring-structured circuits whose cycles cannot embed."""
    phys_size = len(distance_matrix)

    def wkey(u, v):
        return (u, v) if u < v else (v, u)

    pair_gates = []
    for key in sorted(access.keys()):
        qs = access[key]
        if len(qs) == 2 and qs[0] != qs[1]:
            pair_gates.append(tuple(qs))
    C = defaultdict(float)
    W = defaultdict(float)
    nbrs = defaultdict(set)
    t = defaultdict(int)
    for r, (u, v) in enumerate(pair_gates):
        k = wkey(u, v)
        C[k] += 1
        W[k] += 1.0 / (1 + t[u] + t[v])
        t[u] += 1
        t[v] += 1
        nbrs[u].add(v)
        nbrs[v].add(u)
    if not C:
        return None
    act = sorted(nbrs)
    parent = {q: q for q in act}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    tn = defaultdict(set)
    tw = {}
    for (u, v) in sorted(C, key=lambda k: -C[k]):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tn[u].add(v)
            tn[v].add(u)
            tw[wkey(u, v)] = C[wkey(u, v)]
    emb = _subgraph_embed(tn, defaultdict(float, tw), act, backend,
                          phys_size, 300000)
    if emb is None:
        return None
    pos = dict(emb)
    occ = set(pos.values())

    def vc(q, p, skip=None):
        c = 0.0
        for v in nbrs[q]:
            if v == skip or v not in pos:
                continue
            c += W[wkey(q, v)] * (distance_matrix[p][pos[v]] - 1)
        return c
    qs = sorted(pos)
    for _ in range(len(qs) + 1):
        imp = False
        free = [p for p in range(phys_size) if p not in occ]
        for q in qs:
            cur = vc(q, pos[q])
            if cur == 0:
                continue
            bp, bc = None, cur
            for p in free:
                c = vc(q, p)
                if c < bc - 1e-12:
                    bc, bp = c, p
            if bp is not None:
                occ.discard(pos[q])
                occ.add(bp)
                free.remove(bp)
                free.append(pos[q])
                pos[q] = bp
                imp = True
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                qi, qj = qs[i], qs[j]
                if distance_matrix[pos[qi]][pos[qj]] > 2:
                    continue
                if (vc(qi, pos[qj], skip=qj) + vc(qj, pos[qi], skip=qi)
                        < vc(qi, pos[qi], skip=qj)
                        + vc(qj, pos[qj], skip=qi) - 1e-12):
                    pos[qi], pos[qj] = pos[qj], pos[qi]
                    imp = True
        if not imp:
            break
    new_map = [-1] * num_qubits
    used = set()
    for lq, pq in pos.items():
        if 0 <= lq < num_qubits:
            new_map[lq] = pq
            used.add(pq)
    for lq in range(num_qubits):
        if new_map[lq] != -1:
            continue
        for p in range(phys_size):
            if p not in used:
                new_map[lq] = p
                used.add(p)
                break
    return new_map


def generate_spectral_descent_mapping(access, distance_matrix, backend,
                                      num_qubits):
    """Mesh-regime candidate: spectral co-embedding of the logical and
    hardware graphs followed by the standard exchange descent on raw
    weighted stretch. Intended for near-embeddable 2D-mesh interaction
    graphs (e.g. Sycamore-derived QUEKO) on lattice devices, where the
    two spectral geometries align. Returns a full mapping or None."""
    weights, nbrs = _interaction_graph(access)
    if not weights:
        return None
    active = sorted(nbrs.keys())
    phys_size = len(distance_matrix)

    def wkey(u, v):
        return (u, v) if u < v else (v, u)
    try:
        ms, _ = _spectral_coembed(weights, nbrs, active, backend,
                                  phys_size, num_qubits, ndim=2)
    except Exception:
        return None
    W = defaultdict(float)
    for key in sorted(access.keys()):
        qs = access[key]
        if len(qs) == 2 and qs[0] != qs[1]:
            W[wkey(qs[0], qs[1])] += 1.0
    pos = {q: ms[q] for q in active if 0 <= ms[q] < phys_size}
    occ = set(pos.values())

    def vc(q, p, skip=None):
        c = 0.0
        for v in nbrs[q]:
            if v == skip or v not in pos:
                continue
            c += W[wkey(q, v)] * (distance_matrix[p][pos[v]] - 1)
        return c
    qs = sorted(pos)
    for _ in range(len(qs) + 1):
        imp = False
        free = [p for p in range(phys_size) if p not in occ]
        for q in qs:
            cur = vc(q, pos[q])
            if cur == 0:
                continue
            bp, bc = None, cur
            for p in free:
                c = vc(q, p)
                if c < bc - 1e-12:
                    bc, bp = c, p
            if bp is not None:
                occ.discard(pos[q])
                occ.add(bp)
                free.remove(bp)
                free.append(pos[q])
                pos[q] = bp
                imp = True
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                qi, qj = qs[i], qs[j]
                if distance_matrix[pos[qi]][pos[qj]] > 2:
                    continue
                if (vc(qi, pos[qj], skip=qj) + vc(qj, pos[qi], skip=qi)
                        < vc(qi, pos[qi], skip=qj)
                        + vc(qj, pos[qj], skip=qi) - 1e-12):
                    pos[qi], pos[qj] = pos[qj], pos[qi]
                    imp = True
        if not imp:
            break
    new_map = [-1] * num_qubits
    used = set()
    for lq, pq in pos.items():
        if 0 <= lq < num_qubits:
            new_map[lq] = pq
            used.add(pq)
    for lq in range(num_qubits):
        if new_map[lq] != -1:
            continue
        for p in range(phys_size):
            if p not in used:
                new_map[lq] = p
                used.add(p)
                break
    return new_map


def is_mesh_regime(access, backend, phys_size):
    """True for near-embeddable cyclic mesh interaction graphs: every
    logical degree within the device degree bound, a large register and
    a rich cycle structure (>= n/4 independent cycles). Hub circuits
    (any degree above the device bound) and small registers are
    excluded, so the existing pipeline handles them unchanged."""
    nbrs = defaultdict(set)
    edges = set()
    for qs in access.values():
        if len(qs) == 2 and qs[0] != qs[1]:
            u, v = qs
            nbrs[u].add(v)
            nbrs[v].add(u)
            edges.add((u, v) if u < v else (v, u))
    n = len(nbrs)
    if n <= 30 or not edges:
        return False
    hw_max = max(len(backend[p]) for p in backend)
    if any(len(nbrs[q]) > hw_max for q in nbrs):
        return False
    cyclo = len(edges) - n + 1
    return cyclo >= n / 4.0
