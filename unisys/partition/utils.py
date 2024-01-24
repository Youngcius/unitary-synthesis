import networkx as nx
from typing import List
from functools import reduce
from operator import add
from typing import Union
from unisys.basic.gate import Gate
from unisys.basic.circuit import Circuit
from rich.console import Console

console = Console()


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
    print()
    console.print('before unified:')
    console.print(blocks)

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


def block_score(block: Union[List[Gate], Circuit, List[Circuit]], init_score: float = 0) -> float:
    if isinstance(block, Circuit):
        circ = block
    elif isinstance(block, list) and isinstance(block[0], Gate):
        circ = Circuit(block)
    elif isinstance(block, list) and isinstance(block[0], Circuit):
        circ = reduce(add, block)
    else:
        raise ValueError('Invalid type of block.')
    score_local = 0.1 * (circ.num_gates - circ.num_nonlocal_gates)
    score_nl = circ.num_nonlocal_gates  # the number of nonlocal gates contributes the most
    # return init_score + score_nl + 0.1 * score_local
    return score_nl


def contract_1q_gates_on_dag(dag: nx.DiGraph) -> nx.DiGraph:
    """
    Aggregate all 1Q gates into neighboring 2Q gates
    After this pass, each node in DAG is a 2Q block (Circuit instance), including only one 2Q gate
    """
    dag = dag.copy()
    nodes_2q_gate = [node for node in dag.nodes if isinstance(node, Gate) and node.num_qregs == 2]
    for g in nodes_2q_gate:
        block = Circuit([g])
        dag = nx.relabel_nodes(dag, {g: block})
        while True:
            predecessors_1q_gate = [g_nb for g_nb in list(dag.predecessors(block)) if
                                    isinstance(g_nb, Gate) and g_nb.num_qregs == 1]
            successors_1q_gate = [g_nb for g_nb in list(dag.successors(block)) if
                                  isinstance(g_nb, Gate) and g_nb.num_qregs == 1]
            if not predecessors_1q_gate and not successors_1q_gate:  # there is no 1Q gate in the neighborhood
                break
            for g_pred in predecessors_1q_gate:
                block.insert(0, g_pred)
                dag = nx.contracted_nodes(dag, block, g_pred, self_loops=False)
            for g_succ in successors_1q_gate:
                block.append(g_succ)
                dag = nx.contracted_nodes(dag, block, g_succ, self_loops=False)
    return dag
