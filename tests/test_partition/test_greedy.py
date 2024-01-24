import os
import cirq
import yaml
from unisys import Circuit
from functools import reduce
from operator import add
from unisys.partition.greedy import greedy_partition


benchmark_dpath = '../../benchmarks'
with open(os.path.join(benchmark_dpath, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

fnames = []
for dpath in config['circ_dpaths']:
    dpath = os.path.join(benchmark_dpath, dpath)
    fnames.extend([os.path.join(dpath, fname) for fname in os.listdir(dpath)])


num_gates_skip = 300
num_qubits_skip = 8
grain = 2

from tqdm import tqdm


def test_greedy_partition():
    for fname in tqdm(fnames):

        circ = Circuit.from_qasm(fname=fname)

        if circ.num_gates > num_gates_skip or circ.num_qubits > num_qubits_skip:
            continue

        blocks = greedy_partition(circ, grain)

        circ_merged = reduce(add, blocks)
        cirq.testing.assert_allclose_up_to_global_phase(
            circ.unitary(),
            circ_merged.unitary(),
            atol=1e-6,
        )
