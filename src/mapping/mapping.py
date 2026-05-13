
import math
import random
from collections import defaultdict, deque

import islpy as isl
import networkx as nx
import re

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from mqt.qmap.plugins.qiskit.sc import compile_
from mqt.qmap.sc import Architecture, Method, InitialLayout

from pytket.qasm import circuit_from_qasm_str
from pytket.architecture import Architecture
from pytket.placement import GraphPlacement

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

from src.graph.graph import build_backend_graph, compute_distance_matrix

def generate_random_initial_mapping(num_qubits: int):
    """
    Generate a random mapping from logical qubits to physical qubits, as arrays.
    - mapping[logical] = physical
    - reverse_mapping[physical] = logical
    """
    logical_qubits = list(range(num_qubits))
    physical_qubits = list(range(num_qubits))
    random.shuffle(physical_qubits)

    # Initialize arrays
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for logical_qubit, physical_qubit in zip(logical_qubits, physical_qubits):
        mapping[logical_qubit] = physical_qubit
        reverse_mapping[physical_qubit] = logical_qubit

    return mapping, reverse_mapping


def generate_trivial_initial_mapping(num_qubits: int):
    """
    Generate a trivial mapping from logical qubits to physical qubits (arrays).
    - mapping[logical] = logical
    - reverse_mapping[logical] = logical
    """
    mapping = list(range(num_qubits))          # mapping[i] = i
    reverse_mapping = list(range(num_qubits))  # reverse_mapping[i] = i
    return mapping, reverse_mapping
                                                                                                                                                                                                                                                                                                                                                                                                                                   
                                                                                                                                    
def generate_qmap_initial_mapping(qasm_code, backend_edges, num_qubits):                                                             
    circuit = QuantumCircuit.from_qasm_str(qasm_code)                                                                                
                                                                                                                                    
    edges = {tuple(e) for e in CouplingMap(backend_edges).get_edges()}                                                               
    max_q = max(max(u, v) for u, v in edges) + 1                                                                                     
                                                                                                                                    
    arch = Architecture()
    arch.num_qubits = max_q                                                                                                          
    arch.coupling_map = edges
                                                                                                                                    
    mapped_circuit, results = compile_(
        circuit,                                                                                                                     
        arch=arch,
        method=Method.heuristic,                                                                                                     
        initial_layout=InitialLayout.dynamic,
        post_mapping_optimizations=False,                                                                                            
        add_measurements_to_mapped_circuit=False,                                                                                    
    )                                                                                                                                
                                                                                                                                    
    layout = mapped_circuit.layout.initial_layout                                                                                    
                
    mapping = [-1] * num_qubits                                                                                                      
    reverse_mapping = [-1] * num_qubits
                                                                                                                                    
    for v, p in layout.get_virtual_bits().items():
        if v._register.name == "ancilla":                                                                                            
            continue                                                                                                                 
        logical_idx = v._index
        physical_idx = p                                                                                                             
        if logical_idx < num_qubits and physical_idx < num_qubits:
            mapping[logical_idx] = physical_idx                                                                                      
            reverse_mapping[physical_idx] = logical_idx
                                                                                                                                    
    return mapping, reverse_mapping



def generate_pytket_initial_mapping(qasm_code, backend_edges, num_qubits):
    """
    Use pytket's GraphPlacement to generate an initial layout.
    Returns:
    - mapping[logical] = physical
    - reverse_mapping[physical] = logical
    """

    circuit = circuit_from_qasm_str(qasm_code)

    # Architecture needs a sequence, not a set
    edges = list(backend_edges)

    # If your edges are flat integer pairs, this is enough:
    architecture = Architecture(edges)

    placer = GraphPlacement(architecture)
    placement_map = placer.get_placement_map(circuit)

    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for logical_qb, physical_node in placement_map.items():
        if logical_qb.reg_name == "ancilla":
            continue

        logical_idx = logical_qb.index[0]
        physical_idx = physical_node.index[0]

        if 0 <= logical_idx < num_qubits and 0 <= physical_idx < num_qubits:
            mapping[logical_idx] = physical_idx
            reverse_mapping[physical_idx] = logical_idx

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
    """
    Use Qiskit's SabreLayout to generate an initial layout, returned as arrays.
    - mapping[logical] = physical
    - reverse_mapping[physical] = logical
    """
    circuit = QuantumCircuit.from_qasm_str(qasm_code)
    dag_circuit = circuit_to_dag(circuit)
    coupling_map = CouplingMap(backend_edges)
    sabre_layout = SabreLayout(coupling_map, seed=21)
    sabre_layout.run(dag_circuit)

    layout = sabre_layout.property_set["layout"]

    # Figure out how many qubits are in use (excluding ancillas).
    # You could also just assume 'num_qubits = circuit.num_qubits'.
    # For safety, we go by the max qubit index found in the layout.
    max_index = -1
    for v in layout._v2p:
        if v._register.name != "ancilla":
            if v._index > max_index:
                max_index = v._index
            if layout._v2p[v] > max_index:
                max_index = layout._v2p[v]

    # Initialize array-based mappings
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    for v in layout._v2p:
        # Skip ancilla qubits
        if v._register.name == "ancilla":
            continue

        logical_idx = v._index
        physical_idx = layout._v2p[v]
        if logical_idx < num_qubits and physical_idx < num_qubits:
            mapping[logical_idx] = physical_idx
            reverse_mapping[physical_idx] = logical_idx

    return mapping, reverse_mapping


def swap_logical_physical_mappings(logical_to_physical, physical_to_logical, swap_pair, inplace=False):
    """
    Swap the mappings between two physical qubits and update the corresponding logical mappings.
    This function performs a swap operation between two physical qubits by updating both the
    logical-to-physical and physical-to-logical mapping dictionaries/lists. The swap updates
    which logical qubits are mapped to which physical qubits after the swap operation.
    Args:
        logical_to_physical (list): A list where index represents logical qubit and value 
                                   represents the physical qubit it's mapped to
        physical_to_logical (list): A list where index represents physical qubit and value 
                                   represents the logical qubit mapped to it (-1 if unused)
        swap_pair (tuple): A tuple of two integers representing the physical qubits to swap
        inplace (bool, optional): If True, modifies the original mappings in place. 
                                 If False, returns a copy. Defaults to False.
    Returns:
        list: Updated logical_to_physical mapping after the swap operation. If inplace=True,
              this is the same object as the input; otherwise, it's a copy.
    Note:
        When inplace=True, the physical_to_logical mapping is also updated in place.
        The function handles cases where physical qubits may not have logical qubits
        mapped to them (indicated by -1 in physical_to_logical).
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
    """
    Swap the mappings between two physical qubits using ISL (Integer Set Library) operations.

    This function performs a swap operation on an ISL mapping by creating a swap transformation
    that exchanges the mappings for two specified physical qubits.

    Args:
        isl_mapping: An ISL Map object representing the current logical-to-physical qubit mapping
        swap_pair (tuple): A tuple of two integers representing the physical qubits to swap

    Returns:
        An updated ISL Map object with the swap operation applied

    Example:
        If the original mapping has logical qubit 0 -> physical qubit 1 and logical qubit 1 -> physical qubit 2,
        and swap_pair is (1, 2), the result will have logical qubit 0 -> physical qubit 2 and 
        logical qubit 1 -> physical qubit 1.
    """
    q1, q2 = swap_pair

    # Create a set containing the two physical qubits to be swapped
    swap_domain = isl.Set(f"{{[{q1}];[{q2}]}}")

    # Create a mapping that swaps q1 -> q2 and q2 -> q1
    swap_map = isl.Map(f"{{[{q1}] -> [{q2}]; [{q2}] -> [{q1}]}}")

    # Extract mappings that don't involve the qubits being swapped
    other_mapping = isl_mapping.subtract_range(swap_domain)

    # Apply the swap transformation and combine with unchanged mappings
    return isl_mapping.apply_range(swap_map).union(other_mapping)


def swap_logical_physical_isl_mapping_path(isl_mapping, swap_path_map):
    """
    Apply a swap path mapping to transform logical-physical qubit mappings.

    This function updates the ISL mapping by applying swap operations defined in the
    swap_path_map. It handles the transformation of qubit mappings when swaps are
    performed during quantum circuit execution.

    Args:
        isl_mapping: The current ISL (Integer Set Library) mapping representing
                    the logical to physical qubit relationships.
        swap_path_map: A mapping that defines the swap operations to be applied.
                      If empty, no transformation is performed.

    Returns:
        The updated ISL mapping after applying the swap path transformations.
        If swap_path_map is empty, returns the original isl_mapping unchanged.

    Note:
        The function preserves mappings outside the swap domain by subtracting
        the swap domain from the original mapping and then unioning it with
        the transformed mapping.
    """
    if swap_path_map.is_empty():
        return isl_mapping
    other_mapping = isl_mapping.subtract_range(swap_path_map.domain())
    return isl_mapping.apply_range(swap_path_map).union(other_mapping)


def _build_asap_gate_depths(access):
    """ASAP-schedule each gate over its read qubits and return {gate: depth}.
    Depth 1 is the earliest layer; gates with no prior gate on any of their
    qubits land there. The depth is independent of routing/coupling and only
    captures dependency timing on the logical circuit."""
    gate_depth = {}
    last_layer_for_qubit = {}
    for g in sorted(access.keys()):
        qubits = access[g]
        if not qubits:
            continue
        prev = 0
        for q in qubits:
            lp = last_layer_for_qubit.get(q, 0)
            if lp > prev:
                prev = lp
        d = prev + 1
        gate_depth[g] = d
        for q in qubits:
            last_layer_for_qubit[q] = d
    return gate_depth


def _build_time_decayed_interaction_graph(access, gate_depth, tau):
    """Build the symmetric Time-decayed Interaction Graph W on logical qubits.

    W[u][v] = sum over 2q gates g acting on (u, v) of exp(-depth(g)/tau).

    Heavier weight on early gates pushes those interacting qubits to be
    placed adjacent in the initial mapping, since early-gate cost cannot be
    amortized away by later routing decisions.
    """
    interaction = defaultdict(lambda: defaultdict(float))
    logical_qubits = set()
    for g, qubits in access.items():
        if len(qubits) != 2:
            if qubits:
                logical_qubits.add(qubits[0])
            continue
        u, v = qubits
        logical_qubits.add(u)
        logical_qubits.add(v)
        if u == v:
            continue
        d = gate_depth.get(g, 1)
        w = math.exp(-d / tau) if tau > 0 else 1.0
        interaction[u][v] += w
        interaction[v][u] += w
    return interaction, logical_qubits


def _harmonic_centrality(distance_matrix, valid_nodes):
    """Sum 1/(1+d(p,q)) over all reachable q in valid_nodes.

    A topology-aware preference score that rewards physical qubits in dense,
    well-connected regions of the coupling map.
    """
    n = len(distance_matrix)
    centrality = [0.0] * n
    for p in valid_nodes:
        s = 0.0
        row = distance_matrix[p]
        for q in valid_nodes:
            d = row[q]
            if d != float('inf'):
                s += 1.0 / (1.0 + d)
        centrality[p] = s
    return centrality


def _logical_placement_order(interaction, seed_logical):
    """BFS over the logical-interaction graph from seed; orders qubits by
    distance, ties broken by attraction to already-emitted prefix."""
    order = [seed_logical]
    in_order = {seed_logical}
    frontier = list(interaction.get(seed_logical, {}).keys())
    while frontier:
        frontier = [u for u in frontier if u not in in_order]
        if not frontier:
            break
        frontier.sort(key=lambda u: (-sum(interaction[u].get(v, 0.0)
                                          for v in in_order),
                                     -len(interaction.get(u, {})), u))
        nxt = frontier[0]
        order.append(nxt)
        in_order.add(nxt)
        frontier = list(frontier[1:]) + list(interaction.get(nxt, {}).keys())
    for u in interaction:
        if u not in in_order:
            order.append(u)
            in_order.add(u)
    return order


def _score_partial(log2phys, interaction, distance_matrix):
    """Lower-is-better: (unsat_edges, total_weighted_distance)."""
    unsat = 0
    total_d = 0.0
    seen = set()
    for u, p in log2phys.items():
        for v, w in interaction[u].items():
            if v not in log2phys:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            d = distance_matrix[p][log2phys[v]]
            if d == float('inf'):
                unsat += 1
                total_d += 1000.0 * w
                continue
            if d > 1:
                unsat += 1
            total_d += w * d
    return unsat, total_d


def _beam_embed(interaction, distance_matrix, backend_graph, valid_phys,
                centrality, seed_logical, seed_physical, order, beam_size, beta):
    """Beam search over the embedding tree.

    State: (log2phys dict, used_phys frozenset).
    At each step, for the next logical qubit l, expand each beam state by
    trying up to ``beam_size`` candidate physical positions ranked by the
    *partial-embedding* score that would result. Keep the top ``beam_size``
    states overall.

    Single-pass: no routing is ever simulated, only embedding states scored.
    """
    initial_state = ({seed_logical: seed_physical}, frozenset({seed_physical}))
    beams = [initial_state]

    placement_order = [u for u in order if u != seed_logical]

    for l in placement_order:
        new_beams = []
        for log2phys, used_phys in beams:
            placed_neighbors = [(v, log2phys[v]) for v in log2phys
                                if interaction[l].get(v, 0.0) > 0.0]

            union_candidates = set()
            if placed_neighbors:
                for v, pv in placed_neighbors:
                    union_candidates |= (set(backend_graph[pv]) - used_phys)
            if not union_candidates:
                union_candidates = set(p for p in valid_phys
                                       if p not in used_phys)
            if not union_candidates:
                continue

            def candidate_score(p):
                unsat = 0
                wd = 0.0
                for v, pv in placed_neighbors:
                    d = distance_matrix[p][pv]
                    if d == float('inf'):
                        return (10 ** 9, 0.0, 0.0)
                    if d > 1:
                        unsat += 1
                    wd += interaction[l].get(v, 0.0) * d
                free_nbrs_p = len(set(backend_graph[p]) - used_phys - {p})
                return (unsat, wd, -free_nbrs_p - beta * centrality[p])

            ranked = sorted(union_candidates, key=candidate_score)[:beam_size]
            for p in ranked:
                new_l2p = dict(log2phys)
                new_l2p[l] = p
                new_used = used_phys | {p}
                new_beams.append((new_l2p, new_used))

        if not new_beams:
            break

        new_beams.sort(key=lambda s: _score_partial(
            s[0], interaction, distance_matrix))
        beams = new_beams[:beam_size]

    if not beams:
        return {seed_logical: seed_physical}, (0, 0.0)
    best_state = min(beams, key=lambda s: _score_partial(
        s[0], interaction, distance_matrix))
    return best_state[0], _score_partial(best_state[0], interaction,
                                          distance_matrix)


def generate_tig_initial_mapping(access, backend_edges, num_qubits,
                                 tau_frac=1.0 / 3.0, beta=0.05,
                                 num_seeds=12, beam_size=12):
    """Predictive, single-pass initial mapping (TIG-BFS).

    Pipeline (no routing pre-run, no backward pass):
      1. ASAP-depth each 2q gate on the logical-circuit DAG.
      2. Build a Time-decayed Interaction Graph W on logical qubits where
         early gates dominate (exp(-depth/tau) weighting).
      3. Score physical qubits by harmonic centrality on the coupling graph.
      4. Run a small fan of independent greedy BFS embeddings from the
         top-k seed physical qubits and pick the embedding with the most
         literal-adjacency satisfied edges. Each greedy pass anchors the
         highest-mass logical qubit at a seed physical qubit then expands
         to physical positions that simultaneously satisfy the most
         already-placed weighted edges (#unsatisfied is the lexicographic
         primary key, weighted routing distance is the tie-breaker, so a
         subgraph isomorphism is recovered exactly whenever one exists).

    Returns (mapping, reverse_mapping) array pair of length num_qubits.
    """
    backend_graph = build_backend_graph(backend_edges)
    distance_matrix = compute_distance_matrix(backend_graph)
    valid_phys = sorted(backend_graph.keys())

    gate_depth = _build_asap_gate_depths(access)
    max_depth = max(gate_depth.values()) if gate_depth else 1
    tau = max(max_depth * tau_frac, 1.0)

    interaction, logical_qubits = _build_time_decayed_interaction_graph(
        access, gate_depth, tau)

    centrality = _harmonic_centrality(distance_matrix, valid_phys)
    max_centrality = max(centrality) if centrality else 1.0
    if max_centrality > 0:
        centrality = [c / max_centrality for c in centrality]

    neighbors_of = {u: set(interaction[u].keys()) for u in interaction}

    active_logical = sorted(
        [u for u in logical_qubits if interaction.get(u)],
        key=lambda u: (-len(neighbors_of[u]),
                       -sum(interaction[u].values()), u))

    if not active_logical:
        mapping = list(range(num_qubits))
        reverse_mapping = list(range(num_qubits))
        return mapping, reverse_mapping

    seed_logical_candidates = active_logical[:max(1, min(3, len(active_logical)))]

    seed_candidates = sorted(
        valid_phys,
        key=lambda p: (-len(backend_graph[p]), -centrality[p], p))[:num_seeds]

    best_log2phys = None
    best_metric = None
    schedule = sorted({beam_size, max(beam_size, 32), max(beam_size, 64)})
    found_zero = False
    for bs in schedule:
        for seed_logical in seed_logical_candidates:
            order = _logical_placement_order(interaction, seed_logical)
            for seed_p in seed_candidates:
                log2phys, (unsat, wd) = _beam_embed(
                    interaction, distance_matrix, backend_graph, valid_phys,
                    centrality, seed_logical, seed_p, order, bs, beta)
                metric = (unsat, wd)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_log2phys = log2phys
                    if unsat == 0:
                        found_zero = True
                        break
            if found_zero:
                break
        if found_zero:
            break

    log2phys = best_log2phys

    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits
    for l, p in log2phys.items():
        if 0 <= l < num_qubits and 0 <= p < num_qubits:
            mapping[l] = p
            reverse_mapping[p] = l

    placed_phys = set(p for p in mapping if p != -1)
    unused_phys = [p for p in valid_phys
                   if p < num_qubits and p not in placed_phys]
    for l in range(num_qubits):
        if mapping[l] == -1:
            if not unused_phys:
                break
            p = unused_phys.pop(0)
            mapping[l] = p
            reverse_mapping[p] = l

    return mapping, reverse_mapping