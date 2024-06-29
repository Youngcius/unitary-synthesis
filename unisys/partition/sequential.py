"""
Partitioning algorithm using forward scanning on circuit
"""
import numpy as np
from typing import List, Set
from unisys.basic.gate import Gate
from unisys.basic.circuit import Circuit
from unisys.utils.passes import obtain_front_layer
from unisys.partition.utils import block_score

from rich.console import Console

console = Console()


def sequential_partition(circ: Circuit, grain: int = 2) -> List[Circuit]:
    """
    Partition a list of circuits into groups of grain-qubit blocks (subcircuits) by one-round forward pass.
    ---
    Complexity: O(m*n), m is the number of 2Q gates, n is the number of qubits
    """
    if grain <= 1:
        raise ValueError("grain must be greater than 1.")
    if grain < circ.max_gate_weight:
        raise ValueError("grain must be no less than the maximum gate weight of the circuit.")

    # only consider 2-qubit or multi-qubit nonlocal gates (TODO: delete this line)
    # circ_nl = Circuit([g for g in circ if g.num_qregs > 1])
    circ_nl = circ.clone()  # TODO: rename circ_nl (circ_tmp???)
    blocks = []

    # peel all 1Q gates from the first layer
    first_1q_gates = []
    while first_layer_1q := [g for g in circ_nl.layer()[0] if g.num_qregs == 1]:
        first_1q_gates.extend(first_layer_1q)
        for g_1q in first_layer_1q:
            circ_nl.remove(g_1q)

    while circ_nl:  # for each epoch, select a block with the most nonlocal gates
        front_layer = obtain_front_layer(circ_nl)
        # print()
        # console.print('front_layer (num={}): {}'.format(len(front_layer), front_layer))

        block_candidates = [Circuit([g]) for g in front_layer]
        for i, (g, block) in enumerate(zip(front_layer, block_candidates)):
            circ_nl_peeled = circ_nl.clone()
            circ_nl_peeled.remove(g)
            block_candidates[i] = extend_block_over_circuit(block, circ_nl_peeled, grain)

        scores = [block_score(block) for block in block_candidates]
        # console.print('scores: {}'.format(scores))
        block = block_candidates[np.argmax(scores)]
        # console.print('selected block: {} with qubits {}'.format(block, block.qubits), style='bold green')
        for g in block:
            circ_nl.remove(g)
        blocks.append(block)
        # console.print('current num of blocks: {}'.format(len(blocks)))

    # add 1Q gates from first_1q_gates back to corresponding blocks
    dag = circ.to_dag()
    for g in reversed(first_1q_gates):
        for blk in blocks:
            if list(dag.successors(g))[0] in blk:
                # console.print('adding back 1Q gate {} to {}'.format(g, blk), style='bold green')
                blk.insert(0, g)
                break

    assert sum([blk.num_gates for blk in blocks]) == circ.num_gates, "num_gates mismatch"
    # console.print('num_gates_all_blocks: {}'.format(sum([blk.num_gates for blk in blocks])))
    # console.print('num_gates_circ: {}'.format(len(circ)))
    # console.print('num_gates: {}'.format(circ.num_gates))

    # NOTE: in this algorithm we do not need to unify the blocks since they are already sorted
    return blocks


def extend_block_over_circuit(block: Circuit, circ: Circuit, max_weight: int) -> Circuit:
    """Search applicable gates from the front layer of circ to append the to block"""
    block = block.clone()
    if not circ:
        return block
    dag = circ.to_dag()

    # print()
    # console.print('extending {} with qubits {}'.format(block, block.qubits), style='bold blue')

    while front_layer := obtain_front_layer(dag):
        # let optional_gates just be front_layer for simplicity
        optional_gates = _sort_gates_on_ref_qubits(front_layer, set(block.qubits))
        # console.print('sorting key: {}'.format(sorted([(len(set(g.qregs) - set(block.qubits)),
        # -len(set(g.qregs) & set(block.qubits))) for g in optional_gates])))

        # console.print('>>> optional_gates: {}'.format(optional_gates), style='yellow')
        # console.print('block.qubits: {}, front_layer[0].qregs: {}'.format(block.qubits, front_layer[0].qregs))

        # if the first gate in optional_gates is not applicable, then all the gates in optional_gates are not applicable
        if len(set(block.qubits + optional_gates[0].qregs)) > max_weight:
            break

        for g in optional_gates:
            if len(set(block.qubits + g.qregs)) <= max_weight:
                # console.print('! added gate: {}'.format(g), style='green')
                block.append(g)
                dag.remove_node(g)
            else:
                break

    return block


def _sort_gates_on_ref_qubits(gates: List[Gate], ref_qubits: Set[int]) -> List[Gate]:
    """
    Sort the gates according to the number of qubits overlapped and additional overhead with the given set of qubits
    Sort by: 1) additional overhead (ascending); 2) number of qubits overlapped (descending)
    """
    return sorted(gates, key=lambda g: (len(set(g.qregs) - ref_qubits),
                                        - len(set(g.qregs) & ref_qubits)))
