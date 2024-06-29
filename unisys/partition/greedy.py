"""
Partitioning algorithm using greedy searching on dag
"""
import numpy as np
import networkx as nx
from functools import reduce
from operator import add
from typing import List, Set, Union
from unisys.basic.circuit import Circuit
from unisys.utils.passes import dag_to_layers, contract_1q_gates_on_dag
from unisys.partition.utils import block_score

from rich.console import Console

console = Console()


def greedy_partition(circ: Circuit, grain: int = 2) -> List[Circuit]:
    """
    Partition a list of circuits into groups of grain-qubit blocks (subcircuits) by greedy searching.
    ---
    Complexity: O(m^2), m is the number of 2Q gates
    """
    if grain <= 1:
        raise ValueError("grain must be greater than 1.")
    if grain < circ.max_gate_weight:
        raise ValueError("grain must be no less than the maximum gate weight of the circuit.")

    # 1) aggregate all 1Q gates into neighboring 2Q gates: after this pass, each node in DAG is a 2Q block
    dag = contract_1q_gates_on_dag(circ.to_dag())
    # console.print('dag nodes (before partitioning): {}'.format(dag.nodes), style='bold green')

    #############################
    from collections import Counter

    # 2) contract nodes into the by greedy searching
    nx.set_node_attributes(dag, False, 'resolved')  # this attribute will be True for all nodes in the end

    epoch = 0
    while block_candidates := [block for block in dag.nodes if not dag.nodes[block]['resolved']]:
        epoch += 1
        # print()
        # console.print('EPOCH {} new searching epoch (num_candidates: {})'.format(epoch, len(block_candidates)))
        # console.print('RESOLVED attributes: {}'.format(Counter(nx.get_node_attributes(dag, 'resolved').values())))
        blocks_extended = []
        for block in block_candidates:
            block_ = extend_block_over_dag(block, dag, grain)
            blocks_extended.append(block_)

        scores = [block_score(block_) for block_ in blocks_extended]

        # console.print('scores: {}'.format(scores))

        max_score = max(scores)
        involved_blocks = set()
        for i in np.argsort(scores)[::-1]:  # iterate all maximum scores
            if scores[i] != max_score:
                break

            block = block_candidates[i]
            block_ = blocks_extended[i]
            if set(block_) & involved_blocks:
                continue
            involved_blocks |= set(block_)
            # console.print('iteration {}, num of involved blocks: {}'.format(i, len(involved_blocks)), style='bold yellow')

            nodes_to_contract = [node for node in block_ if node != block]  # block_ 目前类型为 List[Circuit] 用意是获取需要缩并的节点
            new_block = reduce(add, block_)
            # console.print('nodes_to_contract: (num = {})'.format(len(nodes_to_contract)), style='bold green')
            # console.print('new resolved block {} with qubits {}'.format(new_block, new_block.qubits),
            #               style='bold green')
            # print(new_block.to_cirq())

            dag = nx.relabel_nodes(dag, {block: new_block})
            nx.set_node_attributes(dag, {new_block: True}, 'resolved')
            for node in nodes_to_contract:
                dag = nx.contracted_nodes(dag, new_block, node, self_loops=False)

    assert all(list(nx.get_node_attributes(dag, 'resolved').values())), 'Not all nodes are resolved.'

    # construct blocks from DAG
    return list(nx.topological_sort(dag))


def extend_block_over_dag(block: Circuit, dag: nx.DiGraph, max_weight: int) -> List[Circuit]:
    """Search applicable gates from the neighbors of block among this DAG to add them to block"""
    # print()
    # console.print('extending block {} with qubits {}'.format(block, block.qubits), style='bold blue')
    block_ = [block]
    if not dag:
        return block_
    dag = dag.copy()

    from collections import Counter

    def reset_topology_order():
        layers = dag_to_layers(dag)
        for topo_order, layer in enumerate(layers):
            for b in layer:
                nx.set_node_attributes(dag, {b: topo_order}, 'topo_order')

    while True:
        ########################################################
        # 1) find potential neighbors from predecessors

        reset_topology_order()
        ref_qubits = reduce(add, block_).qubits
        # console.print('RESOLVED attributes: {}'.format(Counter(nx.get_node_attributes(dag, 'resolved').values())))

        pred_nb_layer = [b for b in dag.predecessors(block) if
                         dag.nodes[b]['topo_order'] == dag.nodes[block]['topo_order'] - 1 and
                         not dag.nodes[b]['resolved']]
        pred_nb_layer = _sort_blocks_on_ref_qubits(pred_nb_layer, ref_qubits)

        # console.print('pred_nb_layer: (num = {})'.format(len(pred_nb_layer)), style='bold red')
        # for b in pred_nb_layer:
        #     console.print('--------------', dag.nodes[b]['resolved'])
        #     print(b.to_cirq())

        # try to contract nodes from pred_nb_layer
        for b in pred_nb_layer:
            if len(set(ref_qubits + b.qubits)) <= max_weight:
                # console.print('!!!!!! added pred neighbor: {}'.format(b), style='bold red')
                block_.insert(0, b)
                ref_qubits = reduce(add, block_).qubits
                dag = nx.contracted_nodes(dag, block, b, self_loops=False)
            else:
                break

        ########################################################
        # 2) find potential neighbors from successors
        reset_topology_order()
        ref_qubits = reduce(add, block_).qubits
        succ_nb_layer = [b for b in dag.successors(block) if
                         dag.nodes[b]['topo_order'] == dag.nodes[block]['topo_order'] + 1 and
                         not dag.nodes[b]['resolved']]
        succ_nb_layer = _sort_blocks_on_ref_qubits(succ_nb_layer, ref_qubits)
        # console.print('succ_nb_layer: (num = {})'.format(len(succ_nb_layer)), style='bold red')
        # for b in succ_nb_layer:
        #     console.print('--------------', dag.nodes[b]['resolved'])
        #     print(b.to_cirq())

        # try to contract nodes from succ_nb_layer
        for b in succ_nb_layer:
            if len(set(ref_qubits + b.qubits)) <= max_weight:
                # console.print('!!!!!! added succ neighbor: {}'.format(b), style='bold red')
                block_.append(b)
                ref_qubits = reduce(add, block_).qubits
                dag = nx.contracted_nodes(dag, block, b, self_loops=False)
            else:
                break

        if not pred_nb_layer and not succ_nb_layer:
            break

        ref_qubits = reduce(add, block_).qubits
        if all([len(set(ref_qubits + b.qubits)) > max_weight for b in pred_nb_layer + succ_nb_layer]):
            break

    return block_


def _sort_blocks_on_ref_qubits(blocks: List[Circuit], ref_qubits: Union[List[int], Set[int]]) -> List[Circuit]:
    """
    Sort the blocks according to the number of qubits overlapped and additional overhead with the given set of qubits
    Sort by: 1) additional overhead (ascending); 2) number of qubits overlapped (descending)
    """
    ref_qubits = set(ref_qubits)
    return sorted(blocks, key=lambda b: (len(set(b.qubits) - ref_qubits),
                                         - len(set(b.qubits) & ref_qubits)))
