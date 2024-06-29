import networkx as nx
from typing import List, Union
from functools import reduce
from operator import add
from unisys.basic.gate import Gate, UnivGate
from unisys.basic.circuit import Circuit
from unisys.utils.operations import approximately_commutative


def dag_to_layers(dag: nx.DiGraph) -> List[List]:
    """Convert a DAG to a list of layers in topological order"""
    dag = dag.copy()
    layers = []
    while dag.nodes:
        front_layer = obtain_front_layer(dag)
        layers.append(front_layer)
        dag.remove_nodes_from(front_layer)
    return layers


def dag_to_circuit(dag: nx.DiGraph) -> Circuit:
    """Convert a DAG to a Circuit"""
    node_is_block = isinstance(next(iter(dag.nodes())), Circuit)
    if node_is_block:
        return Circuit(reduce(add, list(nx.topological_sort(dag))))
    return Circuit(list(nx.topological_sort(dag)))


def obtain_front_layer(dag_or_circ: Union[Circuit, nx.DiGraph]) -> List[Union[Gate, Circuit]]:
    """
    Obtain front layer (with in_degree == 0) of the DAG.
    Since the node of DAG might be Gate instance or Circuit instance, result is a list of Gate or Circuit.
    """
    if isinstance(dag_or_circ, Circuit):
        dag = dag_or_circ.to_dag()
    else:
        dag = dag_or_circ
    front_layer = []
    for node in dag.nodes:
        if dag.in_degree(node) == 0:
            front_layer.append(node)
    return front_layer


def sort_blocks_on_qregs(blocks: List[Circuit], descend=False) -> List[Circuit]:
    if descend:
        return sorted(blocks, key=lambda b: max([max(blk.qubits) for blk in blocks]))
    else:
        return sorted(blocks, key=lambda b: min([min(blk.qubits) for blk in blocks]))


def unify_blocks(blocks: List[Circuit], circ: Circuit) -> List[Circuit]:
    """
    Unify the blocks (reorder them) according to original gates orders for the given circuit.
    """
    # construct the mapping from node indices to blocks
    # print()
    # console.print('before unified:')
    # console.print(blocks)

    # for blk in blocks:
    #     console.print(blk)
    #     print(blk.to_cirq())

    # contract the nodes in the same block, replace the contracted node by the block in DAG
    dag = circ.to_dag()
    for block in blocks:
        dag = nx.relabel_nodes(dag, {block[0]: block})

        # print('DAG nodes after relabeling', dag.nodes)
        for g in block[1:]:
            dag = nx.contracted_nodes(dag, block, g, self_loops=False)

    # blocks_layers = list(map(sort_blocks_on_qregs, dag_to_layers(dag)))
    # blocks = reduce(add, blocks_layers)
    # return blocks
    return list(nx.topological_sort(dag))


def contract_1q_gates_on_dag(dag: nx.DiGraph) -> nx.DiGraph:
    """
    Aggregate all 1Q gates into neighboring 2Q gates
    After this pass, each node in DAG is a 2Q block (Circuit instance), including only one 2Q gate
    """
    dag = dag.copy()
    nodes_2q_gate = [node for node in dag.nodes() if isinstance(node, Gate) and node.num_qregs == 2]

    # console.print('nodes_2q_gate:', nodes_2q_gate)
    # console.print('nodes_2q_gate (indices):', [node_index(dag, node) for node in nodes_2q_gate])

    for g in nodes_2q_gate:
        # console.print({idx: dag[idx] for idx in dag.node_indices()})
        # console.print('contracting {} with node index {}'.format(g, node_index(dag, g)), style='bold red')
        block = Circuit([g])
        dag = nx.relabel_nodes(dag, {g: block})
        # dag[node_index(dag, g)] = block
        while True:
            predecessors_1q = [g_nb for g_nb in list(dag.predecessors(block)) if
                               isinstance(g_nb, Gate) and g_nb.num_qregs == 1]
            successors_1q = [g_nb for g_nb in list(dag.successors(block)) if
                             isinstance(g_nb, Gate) and g_nb.num_qregs == 1]
            # predecessors_1q = find_predecessors_by_node(dag, node_index(dag, block),
            # lambda node: isinstance(node, Gate) and node.num_qregs == 1)
            # successors_1q = find_successors_by_node(dag, node_index(dag, block),
            # lambda node: isinstance(node, Gate) and node.num_qregs == 1)
            if not predecessors_1q and not successors_1q:  # there is no 1Q gate in the neighborhood
                break
            # console.print('predecessors_1q: {}'.format(predecessors_1q), style='blue')
            # console.print('successors_1q: {}'.format(successors_1q), style='blue')
            for g_pred in predecessors_1q:
                block.insert(0, g_pred)
                dag = nx.contracted_nodes(dag, block, g_pred, self_loops=False)
                # dag.contract_nodes([node_index(dag, block), node_index(dag, g_pred)], block)
            for g_succ in successors_1q:
                block.append(g_succ)
                # dag.contract_nodes([node_index(dag, block), node_index(dag, g_succ)], block)
                dag = nx.contracted_nodes(dag, block, g_succ, self_loops=False)
            # print(block.to_cirq())
    return dag


def blocks_to_dag(blocks: List[Circuit], nested: bool = False) -> nx.DiGraph:
    """
    Convert blocks (a list of Circuit instances) to a DAG in networkx.

    Args:
        blocks: a list of Circuit instances
        nested: default False; if True, each node is also a DAG converted from a Circuit
    """
    blocks = blocks.copy()
    dag = nx.DiGraph()

    block_dag_map = {blk: blk.to_dag() for blk in blocks}
    if nested:
        # dag.add_nodes_from([blk.to_dag() for blk in blocks])
        dag.add_nodes_from(list(block_dag_map.values()))
    else:
        dag.add_nodes_from(blocks)

    while blocks:
        blk = blocks.pop(0)
        qubits = set(blk.qubits)
        for blk_opt in blocks:  # traverse subsequent optional blocks
            qubits_opt = set(blk_opt.qubits)
            if dependent_qubits := qubits_opt & qubits:
                if nested:
                    # dag.add_edge(
                    #     node_index(dag, block_dag_map[blk]),
                    #     node_index(dag, block_dag_map[blk_opt]),
                    #     {'qubits': list(dependent_qubits)})
                    dag.add_edge(block_dag_map[blk], block_dag_map[blk_opt])
                else:
                    # dag.add_edge(
                    # node_index(dag, blk),
                    # node_index(dag, blk_opt),
                    # {'qubits': list(dependent_qubits)})
                    dag.add_edge(blk, blk_opt)
                qubits -= qubits_opt
            if not qubits:
                break

    return dag


def compact_dag(dag: nx.DiGraph, num_2q_threshold: int = 4) -> nx.DiGraph:
    """
    Compact a nested DAG (each of its node is also a DAG converted from a Circuit).
    ---
    Opportunity 1: difference between greedy_partition and quick_partition
    Opportunity 2: numerically equivalence when exchanging order of two SU(4) gates
    """
    # Typically, each node is a 3-qubit block, including only 2-qubit gates

    circ = Circuit(reduce(add, [blk.gates for blk in dag.nodes]))
    gate_dependency = circ.to_dag()
    ##########################################################################
    # Opportunity 1
    nx.set_node_attributes(dag, False, 'resolved')  # this attribute will be True for all nodes in the end
    nx.set_node_attributes(dag,
                           {node: True for node in dag.nodes if node.num_nonlocal_gates < num_2q_threshold},
                           'resolved')
    while unresolved_nodes := [node for node in dag.nodes if not dag.nodes[node]['resolved']]:
        node = sorted(unresolved_nodes, key=lambda node: node.num_nonlocal_gates, reverse=True)[0]
        if pred_nodes := list(dag.predecessors(node)):
            for pred_node in pred_nodes:
                if set(pred_node[-1].qregs).issubset(set(node.qubits)) and all(
                        [(g in node) for g in gate_dependency.successors(pred_node[-1])]):
                    # although the first condition is not necessary, it can help improve running performance
                    node.prepend(pred_node.pop())
                    if pred_node.num_nonlocal_gates < num_2q_threshold:
                        dag.nodes[pred_node]['resolved'] = True
        if succ_nodes := list(dag.successors(node)):
            for succ_node in succ_nodes:
                if set(succ_node[0].qregs).issubset(set(node.qubits)) and all(
                        [(pred in node) for pred in gate_dependency.predecessors(succ_node[0])]):
                    # although the first condition is not necessary, it can help improve running performance
                    node.append(succ_node.pop(0))
                    if succ_node.num_nonlocal_gates < num_2q_threshold:
                        dag.nodes[succ_nodes]['resolved'] = True
        dag.nodes[node]['resolved'] = True
        # nx.set_node_attributes(dag,
        #                        {node: True for node in dag.nodes if num_2q_of_dag(node) < num_2q_threshold},
        #                        'resolved')
        # console.print('Now attributes:', nx.get_node_attributes(dag, 'resolved'))
    ##########################################################################
    # Opportunity 2
    # ! in fact, the opportunity-2 pass can be iteratively applied
    nx.set_node_attributes(dag, False, 'resolved')  # this attribute will be True for all nodes in the end
    nx.set_node_attributes(dag,
                           {node: True for node in dag.nodes if node.num_nonlocal_gates < num_2q_threshold},
                           'resolved')
    while unresolved_nodes := [node for node in dag.nodes if not dag.nodes[node]['resolved']]:
        node = sorted(unresolved_nodes, key=lambda node: node.num_nonlocal_gates, reverse=True)[0]
        if pred_nodes := [node for node in dag.predecessors(node) if node.num_nonlocal_gates <= num_2q_threshold + 1]:
            for pred_node in pred_nodes:
                if pred_node.num_gates > 1 and set(pred_node[-2].qregs).issubset(set(node.qubits)):
                    # although the "subset" condition is not necessary, it can help improve running performance
                    circ_tmp = circ.clone()
                    circ_tmp.remove(pred_node[-1])
                    gate_dependency = circ_tmp.to_dag()
                    if all([(g in node) for g in gate_dependency.successors(pred_node[-2])]):
                        if res := approximately_commutative(pred_node[-2], pred_node[-1]):
                            # u_g = Circuit(pred_node[-2:]).unitary()
                            # v_g = Circuit(res).unitary()
                            # u = (pred_node + node).unitary()
                            # console.print('res:', res)
                            circ.insert(circ.index(pred_node[-2]), res[0])
                            circ.remove(pred_node[-2])
                            circ.insert(circ.index(pred_node[-1]), res[1])
                            circ.remove(pred_node[-1])
                            pred_node.pop()
                            pred_node.pop()
                            pred_node.append(res[0])
                            node.prepend(res[1])
                            if pred_node.num_gates > 1 and set(pred_node[-2].tqs) == set(pred_node[-1].tqs):
                                # fuse two SU(4)s cause they act on the same qubits
                                qubits = sorted(pred_node[-1].tqs)
                                su4 = UnivGate(Circuit([pred_node[-2], pred_node[-1]]).unitary()).on(qubits)
                                circ.insert(circ.index(pred_node[-2]), su4)
                                circ.remove(pred_node[-2])
                                circ.remove(pred_node[-1])
                                pred_node.pop()
                                pred_node.pop()
                                pred_node.append(su4)
                            if pred_node.num_nonlocal_gates < num_2q_threshold:
                                dag.nodes[pred_node]['resolved'] = True
        if succ_nodes := [node for node in dag.successors(node) if node.num_nonlocal_gates <= num_2q_threshold + 1]:
            for succ_node in succ_nodes:
                if succ_node.num_gates > 1 and set(succ_node[1].qregs).issubset(set(node.qubits)):
                    # although the "subset" condition is not necessary, it can help improve running performance
                    circ_tmp = circ.clone()
                    circ_tmp.remove(succ_node[0])
                    gate_dependency = circ_tmp.to_dag()
                    if all([(pred in node) for pred in gate_dependency.predecessors(succ_node[1])]):
                        if res := approximately_commutative(succ_node[0], succ_node[1]):
                            circ.insert(circ.index(succ_node[0]), res[0])
                            circ.remove(succ_node[0])
                            circ.insert(circ.index(succ_node[1]), res[1])
                            circ.remove(succ_node[1])
                            succ_node.pop(0)
                            succ_node.pop(0)
                            succ_node.prepend(res[1])
                            node.append(res[0])
                            if succ_node.num_gates > 1 and set(succ_node[0].tqs) == set(succ_node[1].tqs):
                                # fuse two SU(4)s cause they act on the same qubits
                                qubits = sorted(succ_node[0].tqs)
                                su4 = UnivGate(Circuit([succ_node[0], succ_node[1]]).unitary()).on(qubits)
                                circ.insert(circ.index(succ_node[0]), su4)
                                circ.remove(succ_node[0])
                                circ.remove(succ_node[1])
                                succ_node.pop(0)
                                succ_node.pop(0)
                                succ_node.prepend(su4)
                            if succ_node.num_nonlocal_gates < num_2q_threshold:
                                dag.nodes[succ_node]['resolved'] = True
        dag.nodes[node]['resolved'] = True

    return dag

# def pauli_simp(circ: Circuit) -> Circuit:
#     """
#     Apply pytket.passes.PauliSimp to the circuit.
#     """
#     from pytket.passes import PauliSimp
#     from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

#     circ_tket = circuit_from_qasm_str(circ.to_qasm())
#     PauliSimp().apply(circ_tket)
#     return Circuit.from_qasm(circuit_to_qasm_str(circ_tket))
