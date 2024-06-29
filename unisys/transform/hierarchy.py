"""Hierarchical compilation (partitioning + approximate synthesis)"""
import networkx as nx
from unisys.basic.circuit import Circuit
from unisys.transform.approximate import approx_to_su4
from unisys.partition.sequential import sequential_partition
from unisys.transform.unroller import unroll_su4, unroll_tensor_product
from unisys.transform.fuser import fuse_blocks, fuse_neighbor_u3, fuse_u3_to_su4
from unisys.utils import passes
from operator import add
from functools import reduce
from rich.console import Console

console = Console()

# make the threshold as lower-bound

# CNOT_APPROX_THRESHOLD = {
#     3: 14,
#     4: 61,
# }

# SU4_APPROX_THRESHOLD = {
#     3: 6,
#     4: 27,
# }

# make an aggresive threshold for 3-qubit blocks

CNOT_APPROX_THRESHOLD = {
    3: 12,
    4: 58,
}

SU4_APPROX_THRESHOLD = {
    3: 4,
    4: 25,
}



def hierarchical_synthesize(circ: Circuit, grain: int = 3):
    """
    Hierarchical synthesis only involves SU(4) components
    """
    # Step 1: 2-qubit gate fusion
    console.print('Step-1: fusing 2-qubit gates by 2-qubit partitioning', style='bold purple')
    blocks_2q = sequential_partition(circ, 2)
    fused_2q = fuse_blocks(blocks_2q)
    fused_2q = unroll_tensor_product(fused_2q)
    fused_2q = fuse_u3_to_su4(fused_2q)

    # Step 2: partition again and perform approximate synthesis if necessary
    console.print('Step-2: performing approximate synthesis if necessary by grain-qubit partitioning',
                  style='bold purple')
    blocks_3q = sequential_partition(fused_2q, grain)
    dag = passes.blocks_to_dag(blocks_3q)
    dag = passes.compact_dag(dag, num_2q_threshold=SU4_APPROX_THRESHOLD[grain])
    blocks_3q = list(nx.topological_sort(dag))

    num_3q_blocks = len(blocks_3q)
    for i in range(num_3q_blocks):
        if blocks_3q[i].num_nonlocal_gates > SU4_APPROX_THRESHOLD[grain]:
            console.print('There is a {}-#2q {}-qubit block to be approximately synthesized'.format(blocks_3q[i].num_nonlocal_gates, grain), style='bold red')
            approx_block = approx_to_su4(unroll_su4(blocks_3q[i], by='can'), max_synthesis_size=grain)
            if approx_block.num_nonlocal_gates < blocks_3q[i].num_nonlocal_gates:
                blocks_3q[i] = approx_block
                
    # Further step: fuse some 2-qubit gates again by 2-qubit partitioning
    console.print('Further Step: fuse some 2-qubit gates again by 2-qubit partitioning', style='bold purple')
    circ_opt = Circuit(reduce(add, blocks_3q))
    blocks_2q = sequential_partition(circ_opt, 2)
    circ_opt = fuse_blocks(blocks_2q)
    circ_opt = unroll_su4(circ_opt, by='can')

    # Further step: fuse neighboring U3 gates
    console.print('Further Step: fuse neighboring U3 gates', style='bold purple')
    circ_opt = fuse_neighbor_u3(circ_opt)

    return circ_opt
