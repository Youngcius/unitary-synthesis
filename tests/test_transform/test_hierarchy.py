"""Test hierarchical synthesis pass (partitioning + approximate synthesis)"""
import sys

sys.path.append('../..')

from unisys.transform.hierarchy import hierarchical_synthesize
from unisys import Circuit
import cirq


def test_hierarchy_grain_3():
    """
    Test hierarchical synthesis with grain=3 on alu-v0_26.qasm
    """
    circ = Circuit.from_qasm(fname='../../benchmarks/alu/alu-v0_26.qasm')
    circ_opt = hierarchical_synthesize(circ)
    print(circ_opt.to_qiskit().draw(fold=1000))
    circ_opt.to_qiskit().draw('mpl', fold=150, filename='circ_opt.png', style='clifford')
    print('num_gates: {}, num_2q_gates: {}, depth: {}'.format(
        circ.num_gates, circ.num_nonlocal_gates, circ.depth))
    print('num_gates: {}, num_2q_gates: {}, depth: {}'.format(
        circ_opt.num_gates, circ_opt.num_nonlocal_gates, circ_opt.depth))
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        circ_opt.unitary(),
        atol=1e-5
    )


# def test_hierarchy_grain_4():
#     """
#     Test hierarchical synthesis with grain=4 on vqe_uccsd_4.qasm
#     This might consume ~ 60 minutes
#     """
#     circ = Circuit.from_qasm(fname='../../benchmarks/vqe_uccsd/vqe_uccsd_4.qasm')
#     circ_opt = hierarchical_synthesize(circ, grain=4)
#     print(circ_opt.to_qiskit().draw(fold=1000))
#     print('num_gates: {}, num_2q_gates: {}, depth: {}'.format(
#         circ.num_gates, circ.num_nonlocal_gates, circ.depth))
#     print('num_gates: {}, num_2q_gates: {}, depth: {}'.format(
#         circ_opt.num_gates, circ_opt.num_nonlocal_gates, circ_opt.depth))
#     cirq.testing.assert_allclose_up_to_global_phase(
#         circ.unitary(),
#         circ_opt.unitary(),
#         atol=1e-5
#     )
