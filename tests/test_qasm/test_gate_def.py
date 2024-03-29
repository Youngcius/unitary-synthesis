"""Test some customized gate definitions"""
import cirq
import numpy as np
from unisys import Circuit, gate


def test_ryy_by_qiskit():
    from qiskit.quantum_info import Operator
    rad = np.random.random()
    circ = Circuit([gate.RYY(rad).on([0, 1])])
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        Operator(circ.to_qiskit().reverse_bits()).to_matrix(),
        atol=1e-7
    )


def test_can_by_qiskit():
    from qiskit.quantum_info import Operator
    rads = np.random.random(3)
    circ = Circuit([gate.Can(*rads).on([0, 1])])
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        Operator(circ.to_qiskit().reverse_bits()).to_matrix(),
        atol=1e-7
    )
