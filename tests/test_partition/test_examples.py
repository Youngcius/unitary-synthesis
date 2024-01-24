import sys

sys.path.append('../..')

from operator import add
from functools import reduce
from unisys import partition
from unisys import gate, Circuit
import cirq



GRAIN = 3
GREEDY = False

partition_func = partition.greedy_partition if GREEDY else partition.quick_partition

def test_partition_demo():
    circ = Circuit()
    circ.append(gate.X.on(1, 0))
    circ.append(gate.X.on(3, 2))
    circ.append(gate.X.on(2, 0))
    circ.append(gate.X.on(2, 1))
    circ.append(gate.X.on(2, 0))
    circ.append(gate.X.on(3, 0))
    # q_0: ───@───@───────@───@───
    #         │   │       │   │
    # q_1: ───X───┼───@───┼───┼───
    #             │   │   │   │
    # q_2: ───@───X───X───X───┼───
    #         │               │
    # q_3: ───X───────────────X───
    blocks = partition_func(circ, GRAIN)

    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit(reduce(add, blocks)).unitary(),
        circ.unitary(),
        atol=1e-5
    )


def test_partition_alu():
    circ = Circuit.from_qasm(fname='../../benchmarks/demos/alu-v2_33.qasm')
    blocks = partition_func(circ, GRAIN)
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit(reduce(add, blocks)).unitary(),
        circ.unitary(),
        atol=1e-5
    )


def test_partition_adder():
    circ = Circuit.from_qasm(fname='../../benchmarks/demos/adder3.qasm')
    circ = Circuit([g for g in circ if g.num_qregs > 1])
    blocks = partition_func(circ, GRAIN)
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit(reduce(add, blocks)).unitary(),
        circ.unitary(),
        atol=1e-5
    )


def test_partition_qft():
    circ = Circuit.from_qasm(fname='../../benchmarks/qft/qft_8.qasm')
    circ = Circuit([g for g in circ if g.num_qregs > 1])
    blocks = partition_func(circ, GRAIN)
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit(reduce(add, blocks)).unitary(),
        circ.unitary(),
        atol=1e-5
    )


def test_partition_qaoa():
    circ = Circuit.from_qasm(fname='../../benchmarks/qaoa/qaoa_8.qasm')
    circ = Circuit([g for g in circ if g.num_qregs > 1])
    blocks = partition_func(circ, GRAIN)
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit(reduce(add, blocks)).unitary(),
        circ.unitary(),
        atol=1e-5
    )
