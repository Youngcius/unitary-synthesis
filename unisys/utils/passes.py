import networkx as nx
from typing import List, Union

from unisys.basic.gate import Gate
from unisys.basic.circuit import Circuit


def dag_to_layers(dag: nx.DiGraph) -> List[List]:
    """Convert a DAG to a list of layers in topological order"""
    dag = dag.copy()
    layers = []
    while dag.nodes:
        front_layer = obtain_front_layer(dag)
        layers.append(front_layer)
        dag.remove_nodes_from(front_layer)
    return layers


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


def pauli_simp(circ: Circuit) -> Circuit:
    """
    Apply pytket.passes.PauliSimp to the circuit.
    """
    from pytket.passes import PauliSimp
    from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

    circ_tket = circuit_from_qasm_str(circ.to_qasm())
    PauliSimp().apply(circ_tket)
    return Circuit.from_qasm(circuit_to_qasm_str(circ_tket))

