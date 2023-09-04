"""
Heuristic mapping algorithms
---
E.g.,
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from networkx import Graph
from numpy import ndarray
from unisys import Gate, Circuit
from unisys.basic.gate import SWAPGate
from unisys.utils.arch import gene_init_mapping, is_executable, update_mapping, has_decomposed_completely
from unisys.utils.arch import unify_mapped_circuit

from rich.console import Console

console = Console()


def sabre_search(circ: Circuit, device: Graph, num_pass_periods: int = 3) -> Tuple[
    Circuit, Dict[int, int], Dict[int, int]]:
    """
    SABRE heuristic searching algorithm to generate mapping information for a given circuit and device.

    Args:
        circ: Circuit to be mapped
        device: Device representing physical connectivity
        num_pass_periods: Number of pass periods, default is k=3, meaning searching for 2k+1 passes in total

    Returns:
        Circuit whose qregs indices are logical qubits
        Initial mapping from logical to physical qubits
        Final mapping from logical to physical qubits
    """
    init_mapping = None
    for i in range(num_pass_periods):
        console.rule('SABRE bidirectional pass period {}'.format(i))  # 2k+1 passes

        console.print('forward pass', style='bold green')
        mapped_circ, init_mapping, final_mapping = sabre_search_one_pass(circ, device, init_mapping)

        console.print('reversed pass', style='bold blue')
        circ_inv = mapped_circ.inverse()
        init_mapping = final_mapping
        mapped_circ_inv, init_mapping, final_mapping = sabre_search_one_pass(circ_inv, device, init_mapping)

        init_mapping = final_mapping

    mapped_circ, init_mapping, final_mapping = sabre_search_one_pass(circ, device, init_mapping)

    return mapped_circ, init_mapping, final_mapping


def sabre_search_one_pass(circ: Circuit, device: Graph, init_mapping: Dict[int, int] = None,
                          return_circ_with_swaps: bool = False, gene_init_mapping_type: str = 'random') -> Tuple[
    Circuit, Dict[int, int], Dict[int, int]]:
    """
    One pass of SABRE heuristic searching algorithm to generate mapping information for a given circuit and device.

    Args:
        circ: Circuit to be mapped
        device: Device representing physical connectivity
        init_mapping: Initial mapping from logical to physical qubits
        return_circ_with_swaps: Whether to return the circuit with SWAP gates and all intermediate mappings
        gene_init_mapping_type: If init_mapping is None, how to generate the initial mapping
    
    Returns:
        Circuit whose qregs indices are logical qubits
        Initial mapping from logical to physical qubits
        Final mapping from logical to physical qubits
    """
    assert has_decomposed_completely(circ), "The input circuit should be decomposed into 1Q + 2Q gates completely"
    dag = circ.to_dag()
    print('dag nodes:', dag.nodes)
    mappings = []
    decay_params = {q: 0.1 for q in circ.qubits}
    decay_step = 0.001

    # find the front layer first
    front_layer = obtain_front_layer(dag, circ)

    # begin SWAP searching until front_layer is empty
    dist_mat = nx.floyd_warshall_numpy(device)
    if init_mapping is None:
        mapping = gene_init_mapping(circ, device, gene_init_mapping_type)  # temporary (intermediate) mapping
    else:
        mapping = init_mapping
    mappings.append(mapping.copy())
    circ_with_swaps = Circuit()
    exe_gates = []
    scores = {}
    print('initial_mapping:', mapping)
    num_searchs = 0
    while front_layer:
        # print()
        # print('front_layer:', front_layer)

        # reset the exe_gates for each-round searching
        exe_gates.clear()

        # update executive_gates
        for g in front_layer:
            if is_executable(g, mapping, device=device):
                exe_gates.append(g)
        # print('exe_gates:', exe_gates)

        if exe_gates:  # update front_layer and continue the loop
            for g in exe_gates:
                front_layer.remove(g)
                dag.remove_node(circ.index(g))
                circ_with_swaps.append(g)
            front_layer = obtain_front_layer(dag, circ)
        else:  # find suitable SWAP gates
            # reset decay_params each 5 rounds
            num_searchs += 1
            if num_searchs == 5:
                decay_params = {q: 0.1 for q in circ.qubits}
                num_searchs = 0

            scores.clear()
            swap_candidates = obtain_swap_candidates(front_layer, mapping, device)
            # console.print('Swap candidates: {}'.format(swap_candidates))
            for swap in swap_candidates:
                scores[swap] = heuristic_score(front_layer, dag, circ, mapping, swap, dist_mat, decay_params)

            # find the SWAP with minimal score
            idx_min = np.argmin(list(scores.values()))
            swap = swap_candidates[idx_min]
            circ_with_swaps.append(swap)
            decay_params[swap.tqs[0]] += decay_step
            decay_params[swap.tqs[1]] += decay_step
            # print('updated decay params:', decay_params)

            mapping = update_mapping(mapping, swap_candidates[idx_min])
            mappings.append(mapping.copy())
            console.rule('updated: {}, after {}'.format(mapping, swap))
            # print()
    print('final_mapping:', mapping)
    if return_circ_with_swaps:
        return circ_with_swaps, mappings
    return unify_mapped_circuit(circ_with_swaps, mappings), mappings[0], mappings[-1]


def obtain_front_layer(dag: nx.MultiDiGraph, circ: Circuit) -> List[Gate]:
    """Obtain front layer (with in_degree == 0) of the DAG"""
    front_layer = []
    for node in dag.nodes:
        if dag.in_degree(node) == 0:
            front_layer.append(circ[node])
    return front_layer


def obtain_logical_neighbors(qubit: int, mapping: Dict[int, int], device: Graph) -> List[int]:
    """Obtain logical neighbors of a logical qubit according to the logical-physical mapping and device connectivity"""
    # print(qubit, mapping[qubit])
    physical_neighbors = device.neighbors(mapping[qubit])
    inverse_mapping = {v: k for k, v in mapping.items()}
    # NOTE: physical neighbors may not exist in the mapping because num_device_qubits >=  num_logical_qubits
    logical_neighbors = [inverse_mapping[q] for q in physical_neighbors if q in inverse_mapping]
    return logical_neighbors


def obtain_swap_candidates(gates: List[Gate], mapping: Dict[int, int], device: Graph) -> List[SWAPGate]:
    """Obtain SWAP candidates whose involved qubits must occur in involved qubits of gates"""
    swap_candidates = []
    qubits = Circuit(gates).qubits
    for q in qubits:
        logical_neighbors = obtain_logical_neighbors(q, mapping, device)
        swap_candidates.extend([SWAPGate().on([q, nb]) for nb in logical_neighbors])
    return swap_candidates


def heuristic_score(front_layer: List[Gate], dag: nx.MultiDiGraph, circ: Circuit, mapping: Dict[int, int],
                    swap: SWAPGate, dist_mat: ndarray, decay_params: Dict[float, int]) -> float:
    """
    Effect of decay parameter: 
        trade-off between number of gates and depth, i.e., if q is involved in a SWAP recently, then its decay parameter will increase by delta
    """
    succeeding_layer = set()
    succeeding_weight = 0.5
    for g in front_layer:
        for successor in dag.successors(circ.index(g)):
            # print('{} --> {}'.format(g, circ[successor]))
            if len(succeeding_layer) < 20:
                succeeding_layer.add(circ[successor])
    succeeding_layer = list(succeeding_layer)
    mapping = update_mapping(mapping, swap)  # update mapping after an acted SWAP gate
    # NOTE: only consider 2Q gates for calculating the heuristic score
    front_layer_2q = [g for g in front_layer if len(g.qregs) == 2]
    succeeding_layer_2q = [g for g in succeeding_layer if len(g.qregs) == 2]
    # console.rule('debugging heuristic_score')
    # console.print({'front_layer_2q': front_layer_2q, 'succeeding_layer_2q': succeeding_layer_2q})
    s1 = sum([dist_mat[mapping[g.qregs[0]], mapping[g.qregs[1]]] for g in front_layer_2q]) / len(front_layer_2q)
    if succeeding_layer_2q:
        s2 = sum([dist_mat[mapping[g.qregs[0]], mapping[g.qregs[1]]] for g in succeeding_layer_2q]) / len(
            succeeding_layer_2q)
    else:
        s2 = 1
    # print('s1:', s1, 's2:', s2, 'total:',
    #       max(decay_params[swap.tqs[0]], decay_params[swap.tqs[1]]) * (s1 + succeeding_weight * s2))
    # print('decay_params:', decay_params)
    return max(decay_params[swap.tqs[0]], decay_params[swap.tqs[1]]) * (s1 + succeeding_weight * s2)
