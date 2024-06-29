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
