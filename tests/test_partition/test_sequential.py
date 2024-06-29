import os
import cirq
from unisys import Circuit
from functools import reduce
from operator import add
from tqdm import tqdm
from unisys.partition.sequential import sequential_partition

benchmark_dpath = '../../benchmarks'
fnames = []
for root, dirs, files in os.walk(benchmark_dpath):
    for file in files:
        if file.endswith('.qasm'):
            fnames.append(os.path.join(root, file))

num_gates_skip = 300
num_qubits_skip = 8
grain = 2


def test_sequential_partition():
    for fname in tqdm(fnames):

        circ = Circuit.from_qasm(fname=fname)

        if circ.num_gates > num_gates_skip or circ.num_qubits > num_qubits_skip:
            continue

        blocks = sequential_partition(circ, grain)

        circ_merged = reduce(add, blocks)
        cirq.testing.assert_allclose_up_to_global_phase(
            circ.unitary(),
            circ_merged.unitary(),
            atol=1e-6,
        )
