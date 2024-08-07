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
from unisys.utils.arch import gene_init_mapping, is_executable, update_mapping, unify_mapped_circuit
from unisys.utils.arch import obtain_logical_neighbors
from unisys.utils.passes import obtain_front_layer
from rich.console import Console

console = Console()


def sabre_search(circ: Circuit, device: Graph, 
                 num_pass_periods: int = 3,
                 init_mapping: Dict[int, int] = None,
                 gene_init_mapping_type = 'trivial') -> Tuple[Circuit, Dict[int, int], Dict[int, int]]:
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
    mapped_circ, init_mapping, final_mapping = sabre_search_one_pass(circ, device, init_mapping=init_mapping,
                                                                     gene_init_mapping_type=gene_init_mapping_type)
    circ_inv = circ.inverse()

    last_init_mapping, last_final_mapping = init_mapping, final_mapping

    for i in range(num_pass_periods):
        console.rule('SABRE bidirectional pass period {}'.format(i))

        console.print('reversed pass', style='bold blue')
        init_mapping = final_mapping
        _, _, final_mapping = sabre_search_one_pass(circ_inv, device, init_mapping)

        console.print('forward pass', style='bold green')
        init_mapping = final_mapping
        mapped_circ, init_mapping, final_mapping = sabre_search_one_pass(circ, device, init_mapping)

        if init_mapping == last_init_mapping and final_mapping == last_final_mapping:
            console.print('Bidirectional mapping converged!', style='bold red')
            break
        last_init_mapping, last_final_mapping = init_mapping, final_mapping

    return mapped_circ, init_mapping, final_mapping


def sabre_search_one_pass(circ: Circuit, device: Graph, init_mapping: Dict[int, int] = None,
                          return_circ_with_swaps: bool = False, gene_init_mapping_type: str = 'trivial') -> Tuple[Circuit, Dict[int, int], Dict[int, int]]:
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
    assert _has_decomposed_completely(circ), "The input circuit should be decomposed into 1Q + 2Q gates completely"
    dag = circ.to_dag()
    print('dag nodes:', dag.nodes)
    mappings = []
    decay_params = {q: 0.1 for q in circ.qubits}
    decay_step = 0.001

    # find the front layer first
    # front_layer = obtain_front_layer(dag, circ)
    front_layer = obtain_front_layer(dag)

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
    num_searches = 0
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
                # dag.remove_node(circ.index(g))
                dag.remove_node(g)
                circ_with_swaps.append(g)
            front_layer = obtain_front_layer(dag)
        else:  # find suitable SWAP gates
            # reset decay_params each 5 rounds
            num_searches += 1
            if num_searches == 5:
                decay_params = {q: 0.1 for q in circ.qubits}
                num_searches = 0

            scores.clear()
            swap_candidates = obtain_swap_candidates(front_layer, mapping, device)
            # console.print('Swap candidates: {}'.format(swap_candidates))
            for swap in swap_candidates:
                scores[swap] = heuristic_score(front_layer, dag, mapping, swap, dist_mat, decay_params)

            # find the SWAP with minimal score
            swap = swap_candidates[np.argmin(list(scores.values()))]
            circ_with_swaps.append(swap)
            decay_params[swap.tqs[0]] += decay_step
            decay_params[swap.tqs[1]] += decay_step
            # print('updated decay params:', decay_params)

            mapping = update_mapping(mapping, swap)
            mappings.append(mapping.copy())
            console.rule('updated: {}, after {}'.format(mapping, swap))
            # print()
    print('final_mapping:', mapping)
    if return_circ_with_swaps:
        return circ_with_swaps, mappings
    return unify_mapped_circuit(circ_with_swaps, mappings), mappings[0], mappings[-1]


def obtain_swap_candidates(gates: List[Gate], mapping: Dict[int, int], device: Graph) -> List[SWAPGate]:
    """Obtain SWAP candidates whose involved qubits must occur in involved qubits of gates"""
    swap_candidates = []
    qubits = Circuit(gates).qubits
    for q in qubits:
        logical_neighbors = obtain_logical_neighbors(q, mapping, device)
        swap_candidates.extend([SWAPGate().on([q, nb]) for nb in logical_neighbors])
    return swap_candidates


def heuristic_score(front_layer: List[Gate], dag: nx.DiGraph, mapping: Dict[int, int],
                    swap: SWAPGate, dist_mat: ndarray, decay_params: Dict[int, float]) -> float:
    """
    Effect of decay parameter: 
        trade-off between number of gates and depth, i.e., if q is involved in a SWAP recently, then its decay parameter will increase by delta
    """
    succeeding_layer = set()
    succeeding_weight = 0.5
    for g in front_layer:
        for successor in dag.successors(g):
            # print('{} --> {}'.format(g, circ[successor]))
            if len(succeeding_layer) < 20:
                succeeding_layer.add(successor)
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


def _has_decomposed_completely(circ: Circuit):
    """
    Distinguish whether the circuit has been decomposed into "1Q + 2Q" gates completely but without SWAP gates
    """
    for g in circ:
        if len(g.qregs) > 2 or isinstance(g, SWAPGate):
            return False
    return True
