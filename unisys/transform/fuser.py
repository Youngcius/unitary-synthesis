"""
This module includes functions mainly for partitioned blocks
"""
import networkx as nx
from typing import List
from unisys.basic import gate
from unisys.basic.gate import Gate
from unisys.basic.circuit import Circuit
from unisys.utils.operations import params_u3, tensor_1_slot
from unisys.utils.passes import dag_to_circuit
from unisys.utils.graphs import filter_nodes, find_predecessors_by_node, find_successors_by_node


def fuse_blocks(blocks: List[Circuit], name: str = 'U') -> Circuit:
    """
    Fuse each block in the list into a single multi-qubit gate.
    """
    return Circuit([gate.UnivGate(blk.unitary(), name=name).on(blk.qubits) for blk in blocks])


def fuse_u3_to_su4(circ: Circuit) -> Circuit:
    """Contract all single-qubit gates into neighboring SU(4)"""
    dag = circ.to_dag()
    nodes_2q_gate = filter_nodes(dag, lambda node: isinstance(node, Gate) and node.num_qregs == 2)
    for g in nodes_2q_gate:
        while True:
            predecessors_1q = find_predecessors_by_node(dag, g, lambda node: isinstance(node, Gate) and node.num_qregs == 1)
            successors_1q = find_successors_by_node(dag, g, lambda node: isinstance(node, Gate) and node.num_qregs == 1)
            if not predecessors_1q and not successors_1q:  # there is no 1Q gate in the neighborhood
                break
            for g_pred in predecessors_1q:
                if g_pred.tq == g.tqs[0]:
                    g.data = g.data @ tensor_1_slot(g_pred.data, 2, 0)
                else:
                    g.data = g.data @ tensor_1_slot(g_pred.data, 2, 1)
                dag = nx.contracted_nodes(dag, g, g_pred, self_loops=False)
            for g_succ in successors_1q:
                if g_succ.tq == g.tqs[0]:
                    g.data = tensor_1_slot(g_succ.data, 2, 0) @ g.data
                else:
                    g.data = tensor_1_slot(g_succ.data, 2, 1) @ g.data
                dag = nx.contracted_nodes(dag, g, g_succ, self_loops=False)

    return dag_to_circuit(dag)


def fuse_neighbor_u3(circ: Circuit) -> Circuit:
    """Fuse neighboring single-qubit gates into one U3 gate"""
    dag = circ.to_dag()
    nodes_1q_gate = filter_nodes(dag, lambda node: isinstance(node, Gate) and node.num_qregs == 1)
    for g in nodes_1q_gate:
        # if g has been fused, or g has no successor, skip
        if g not in dag.nodes() or not list(dag.successors(g)):
            continue
        succ = next(dag.successors(g))
        if succ.num_qregs != 1:  # skip 2Q neighbor gates
            continue
        # fuse them into a new U3 gate
        u3 = gate.U3(*params_u3(succ.data @ g.data)).on(g.tq)
        dag = nx.contracted_nodes(dag, g, succ, self_loops=False)
        dag = nx.relabel_nodes(dag, {g: u3})

    return dag_to_circuit(dag)
