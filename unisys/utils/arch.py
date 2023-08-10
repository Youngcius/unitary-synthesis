"""
Arch/Compiler-related Utils functions
"""
import random
import numpy as np
import networkx as nx
from numpy import ndarray
from typing import List, Dict, Tuple
from networkx import Graph
from unisys import Gate, Circuit
from unisys import gate
from unisys.basic.gate import SWAPGate


def has_decomposed_completely(circ: Circuit):
    """
    Distinguish whether the circuit has been decomposed into "1Q + 2Q" gates completely but without SWAP gates
    """
    for g in circ:
        if len(g.qregs) > 2 or isinstance(g, SWAPGate):
            return False
    return True


def gene_init_mapping(circ: Circuit, device: Graph, method: str = 'random') -> Dict[int, int]:
    """
    Generate initial mapping
    
    Args:
        circ: input Circuit instance
        device: coupling graph representing qubit connections
        method: method to generate initial mapping ('random' or 'trivial'), default is 'random'

    Returns:
        A dictionary containing a logical to physical qubits mapping
    """
    n = circ.num_qubits
    logical_qubits = circ.qubits  # logical units
    physical_qubits = sorted(device.nodes)  # physical units
    print('-----' * 10)
    print('logical_qubits:', logical_qubits)
    print('physical_qubits:', physical_qubits)
    assert len(physical_qubits) >= len(
        logical_qubits), "The number of physical qubits should be greater than the number of logical qubits"
    if method == 'random':
        # random.seed(1)
        random.shuffle(physical_qubits)
        subgraph = device.subgraph(physical_qubits[:n])
        while not nx.is_connected(subgraph):
            random.shuffle(physical_qubits)
            subgraph = device.subgraph(physical_qubits[:n])
        mapping = dict(zip(logical_qubits, physical_qubits[:n]))
    elif method == 'trivial':
        assert nx.is_connected(device.subgraph(physical_qubits[:n])), "The first n physical qubits should be connected"
        mapping = dict(zip(logical_qubits, physical_qubits[:n]))
    else:
        raise ValueError("Invalid method to generate initial mapping")
    return mapping


def is_executable(gate: Gate, mapping: Dict[int, int], dist_mat: ndarray = None, device: Graph = None):
    """
    Determine if a gate is executable, i.e., whether the gate acts on qubits which are connected in the coupling graph

    Args:
        gate: input gate
        mapping: logical to physical qubit mapping
        dist_mat: distance matrix of the coupling graph (optional)
        device: coupling graph representing qubit connections (optional)
    
    Returns:
        True if the input gate acts on connected qubits. False otherwise.
    """
    if len(gate.qregs) == 1:  # single qubit gate must be executable
        return True
    assert len(gate.qregs) == 2, "Only 1 or 2 qubit gates are supported"
    assert not (dist_mat is None and device is None), "Either dist_mat or device should be provided"
    if dist_mat is not None:
        return dist_mat[mapping[gate.qregs[0]], mapping[gate.qregs[1]]] == 1
    if device is not None:
        return device.has_edge(mapping[gate.qregs[0]], mapping[gate.qregs[1]])


def update_mapping(mapping: Dict[int, int], swap: SWAPGate) -> Dict[int, int]:
    """Update the logical-to-physical qubit mapping when inserting a SWAP gate"""
    updated_mapping = mapping.copy()
    updated_mapping.update({
        swap.qregs[0]: mapping[swap.qregs[1]],
        swap.qregs[1]: mapping[swap.qregs[0]]
    })
    return updated_mapping


def unify_mapped_circuit(circ_with_swaps: Circuit, mappings: List[Dict[int, int]]) -> Circuit:
    """
    Unify the output circuit, i.e., generate the mapped circuit according to the initial mapping and temporary mappings with SWAP gates inserted

    Args:
        circ_with_swaps: input circuit (with inserted SWAP gates), whose qregs indices are logical qubits
        mappings: initial mapping and all temporary (intermediate) mappings after each SWAP gate

    Returns:
        A new circuit (with SWAP gates decomposed into CNOTs), whose qregs indices are still logical qubits with respect to the initial mapping
    """
    mapping_idx = 0
    mapping = mappings[mapping_idx]
    mapped_circ = Circuit()
    for g in circ_with_swaps:
        if isinstance(g, SWAPGate):
            # NOTE: there must be len(mappings) == len(swaps) + 1
            mapped_circ.append(
                gate.X.on(mapping[g.tqs[0]], mapping[g.tqs[1]]),
                gate.X.on(mapping[g.tqs[1]], mapping[g.tqs[0]]),
                gate.X.on(mapping[g.tqs[0]], mapping[g.tqs[1]]),
            )
            mapping_idx += 1
            mapping = mappings[mapping_idx]
        elif len(g.qregs) == 1:
            mapped_circ.append(g.on(mapping[g.tq]))
        elif len(g.qregs) == 2:
            if g.cqs:
                mapped_circ.append(g.on(mapping[g.tq], mapping[g.cq]))
            else:
                mapped_circ.append(g.on([mapping[q] for q in g.tqs]))
        else:
            raise ValueError("Exceptional gate {}".format(g))
    inverse_mapping = {v: k for k, v in mappings[0].items()}  # inverse of the initial mapping
    mapped_circ = mapped_circ.rewire(inverse_mapping)
    return mapped_circ


def verify_mapped_circuit(circ: Circuit, mapped_circ: Circuit, init_mapping: Dict[int, int],
                          final_mapping: Dict[int, int]):
    """
    Verify the correctness of the mapped circuit with respect to the original circuit.
    Fist append necessary SWAP gates on the mapped circuit.
    Then compare the unitary matrices of the two circuits.

    Args:
        circ: original circuit, whose qregs indices are logical qubits
        mapped_circ: mapped circuit, whose qregs indices are logical qubits
        init_mapping: initial mapping from logical to physical qubits
        final_mapping: final mapping from logical to physical qubits

    Returns:
        True if the mapped circuit is correct. False otherwise.
    """
    init_final_mapping = obtain_logic_logic_mapping(init_mapping, final_mapping)
    mat1 = circ.unitary()
    mapped_circ = mapped_circ.clone()
    mapped_circ.append(*obtain_appended_swaps(init_final_mapping))
    mat2 = mapped_circ.unitary()
    return np.allclose(mat1, mat2)


def verify_mapped_circuit_by_state(circ: Circuit, mapped_circ: Circuit, init_mapping: Dict[int, int],
                                   final_mapping: Dict[int, int]):
    # by comparing final states
    s0 = np.random.rand(2 ** circ.num_qubits)  # random initial state
    s0 = s0 / np.linalg.norm(s0)
    s1 = circ.unitary() @ s0
    s2 = mapped_circ.unitary() @ s0
    # operator.tensor_slots()
    return np.allclose(s1, s2)


def obtain_appended_swaps(mapping) -> List[Gate]:
    """Obtain the corresponding SWAP gates to the mapping relationship, by calculating the series of elementary permutations"""
    swaps = []
    permutations = calculate_elementary_permutations(list(mapping.values()), list(mapping.keys()))
    for perm in permutations:
        swaps.append(gate.SWAP.on([perm[0], perm[1]]))
    return swaps


def calculate_elementary_permutations(initial_rank, target_rank) -> List[Tuple[int, int]]:
    """Calculate the series of elementary permutations to transform the initial rank to the target rank"""
    elementary_permutations = []
    while initial_rank != target_rank:
        for i in range(len(initial_rank)):
            if initial_rank[i] != target_rank[i]:
                # find the indices of the two elements that need to be swapped
                index1 = i
                index2 = initial_rank.index(target_rank[i])
                # change two elements
                initial_rank[index1], initial_rank[index2] = initial_rank[index2], initial_rank[index1]
                # add the permutation relationship of the swap to the output list
                elementary_permutations.append((index1, index2))
    return elementary_permutations


def obtain_logic_logic_mapping(mapping1: Dict[int, int], mapping2: Dict[int, int]) -> Dict[int, int]:
    """Obtain the logical-logical mapping from two logical-physical mappings"""
    inverse_mapping1 = {v: k for k, v in mapping1.items()}
    inverse_mapping2 = {v: k for k, v in mapping2.items()}
    physical_qubits = sorted(mapping1.values())
    logic_logic_mapping = {inverse_mapping1[q]: inverse_mapping2[q] for q in physical_qubits}
    logic_logic_mapping = dict(sorted(logic_logic_mapping.items(), key=lambda x: x[0]))
    return logic_logic_mapping


def read_device_topology(fname: str) -> Graph:
    """
    Read device topology from a file (e.g., .graphml file)
    ---
    NOTE: the node labels in the file might be strings, so we need to convert them to integers
    NOTE: the node labels in the file might not be sorted, so we need to sort them
    NOTE: the graph parsed from the file might be DiGraph, so we need to convert it to Graph
    """
    g = nx.read_graphml(fname)
    device = nx.Graph()
    device.add_nodes_from(sorted(g.nodes))
    device.add_edges_from(g.edges)
    device = nx.convert_node_labels_to_integers(device)
    return device


def gene_grid_2d_graph(total_number: int) -> Graph:
    """Generate a 2D grid graph that is most nearly square"""
    # 26: 5x5 + 1
    # 17: 4x4 + 1
    n = int(np.sqrt(total_number))
    m = int(np.ceil(total_number / n))  # m >= n
    # remainder = total_number % n
    g = nx.grid_2d_graph(n, m)
    g = nx.convert_node_labels_to_integers(g)
    return nx.subgraph(g, range(total_number))


def gene_random_circuit(num_qubits: int, depth: int, op_density: float):
    # , gate_domain: Optional[Dict[Gate, int]] = None):
    """Generate a random circuit"""
    # qubits: Union[Sequence[ops.Qid], int],
    # n_moments: int,
    # op_density: float,
    # gate_domain: Optional[Dict[ops.Gate, int]] = None,
    # random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    raise NotImplementedError
