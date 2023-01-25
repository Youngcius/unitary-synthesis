"""
State preparation and measurement algorithms
"""
import numpy as np
from typing import List
from numpy.linalg import norm
from numpy import conjugate, sqrt

from ..basic import Gate, Circuit
from ..basic import gate, circuit
from ..utils.operator import params_u3, params_abc, params_zyz


def arbitrary_2_qubit_state_circuit(state: np.ndarray, tqs: List[int] = None, return_u3: bool = False) -> Circuit:
    """
    By calculating such a circuit with one CZ gate and three 1-qubit gates
    that will transform the input state into |00>, then reverse this circuit to
    acquire target circuit
        ---------@----W2--- |0>
                 |
        ---W1----@----W3--- |0>
    """
    if not np.allclose(norm(state), 1) or state.shape != (4,):
        raise ValueError('The input is not a reasonable 2-qubit state!')
    if tqs is None:
        tqs = [0, 1]  # TODO: designate tqs
    a0, a1, a2, a3 = state
    v1, v2 = state[:2], state[2:]
    if np.allclose(v1 @ v2, 0):

        k = np.linalg.norm(v2) / np.linalg.norm(v1)
    else:
        k = - np.linalg.norm(v2) / np.linalg.norm(v1) * (conjugate(v1) @ v2) / np.linalg.norm(conjugate(v1) @ v2)

    W1 = parametrized_su2(a3 - k * a1, conjugate(a2 - k * a0)).T
    print(W1.round(3))

    psi_1 = gate.CZ.data @ np.kron(np.identity(2), W1) @ state

    b0, b1, b2, b3 = psi_1
    W2 = parametrized_su2(conjugate(b1), conjugate(b3))

    psi_2 = np.kron(W2, np.identity(2)) @ psi_1

    c0, c1, c2, c3 = psi_2
    W3 = parametrized_su2(conjugate(c0), -conjugate(c1)).T

    # assert np.allclose(np.kron(np.identity(2), W3) @ psi_2, [1, 0, 0, 0])

    if return_u3:
        return [
            gate.U3(*params_u3(W2.conj().T), tq=0),
            gate.U3(*params_u3(W3.conj().T), tq=1),
            gate.CZ,
            gate.U3(*params_u3(W1.conj().T), tq=1),

        ]
    else:
        return [
            Gate(W2.conj().T, tq=0),
            Gate(W3.conj().T, tq=1),
            gate.CZ,
            Gate(W1.conj().T, tq=1)
        ]


# def arbitrary_3_qubit_state_circuit(state: np.ndarray, return_u3: bool = False) -> List[Gate]:
#     """
#     TODO: supply this comments
#         ---W1---------------@----@----W6--- |0>
#                             |    |
#         ---W2----@----W4---------@----W7--- |0>
#                  |          |
#         ---W3----@----W5----@---------W8--- |0>
#     """
#
#     if not np.allclose(norm(state), 1) or state.shape != (8,):
#         raise ValueError('The input is not a reasonable 3-qubit state!')
#
#     if return_u3:
#         circ = [
#             gate.Gate(W1, tq=0),
#             gate.Gate(W2, tq=1),
#             gate.Gate(W3, tq=2),
#             gate.CZ.set_qregs(tq=2, cq=1),
#             gate.Gate(W4, tq=1),
#             gate.Gate(W5, tq=2),
#             gate.CZ.set_qregs(tq=2, cq=0),
#             gate.CZ.set_qregs(tq=1, cq=0),
#             gate.Gate(W6, tq=0),
#             gate.Gate(W7, tq=1),
#             gate.Gate(W8, tq=2),
#         ]
#     else:
#         circ = [
#             gate.U3(*params_u3(W1), tq=0),
#             gate.U3(*params_u3(W2), tq=1),
#             gate.U3(*params_u3(W3), tq=2),
#             gate.CZ.set_qregs(tq=2, cq=1),
#             gate.U3(*params_u3(W4), tq=1),
#             gate.U3(*params_u3(W5), tq=2),
#             gate.CZ.set_qregs(tq=2, cq=0),
#             gate.CZ.set_qregs(tq=1, cq=0),
#             gate.U3(*params_u3(W6), tq=0),
#             gate.U3(*params_u3(W7), tq=1),
#             gate.U3(*params_u3(W8), tq=2),
#         ]
#
#     return circuit.inverse_circuit(circ)
#

def parametrized_su2(x, y) -> np.ndarray:
    """
    Return such a parametrized unitary operator belonging to SU(2)
    :math:
        `U\left( x,y \right) =\frac{1}{\sqrt{|x|^2+|y|^2}}\left( \begin{matrix}
            x&		y\\
            -y^*&		x^*\\
        \end{matrix} \right)`
    """
    return np.array([
        [x, y],
        [-conjugate(y), conjugate(x)]
    ]) / np.linalg.norm([x, y])


if __name__ == '__main__':
    ##############################################
    # example 1: arbitrary 2-qubit state
    psi = np.array([1j, -2, -2 + 1j, 2 - 1j]) / sqrt(15)
    circ = arbitrary_2_qubit_state_circuit(psi, return_u3=True)
    for g in circ:
        print(g.data)

    circuit.disp_circuit(circ)

    ##############################################
    # example 2: arbitrary 3-qubit state
