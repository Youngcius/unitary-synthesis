"""
This module includes functions mainly for partitioned blocks
"""
from unisys.basic.circuit import Circuit
from unisys.basic import gate
from unisys.utils.operations import params_u3
from typing import List
import networkx as nx


def fuse_blocks(blocks: List[Circuit], name: str = 'U') -> Circuit:
    """
    Fuse the blocks in the list into a single multi-qubit gate.
    """
    return Circuit([gate.UnivGate(blk.unitary(), name=name).on(blk.qubits) for blk in blocks])


def fuse_neighbor_u3(circ: Circuit) -> Circuit:
    """Fuse neighboring single-qubit gates into one U3 gate"""
    dag = circ.to_dag()
    nodes_1q_gate = [node for node in dag.nodes if isinstance(node, gate.Gate) and node.num_qregs == 1]
    for g in nodes_1q_gate:
        # if g has been fused, or g has no successor, skip
        if g not in dag.nodes or not list(dag.successors(g)):
            continue
        succ = next(dag.successors(g))
        if succ.num_qregs != 1:  # skip 2Q neighbor gates
            continue
        # fuse them into a new U3 gate
        u3 = gate.U3(*params_u3(succ.data @ g.data)).on(g.tq)
        dag = nx.contracted_nodes(dag, g, succ, self_loops=False)
        dag = nx.relabel_nodes(dag, {g: u3})

    return Circuit(list(nx.topological_sort(dag)))
