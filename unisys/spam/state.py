"""
State preparation and measurement algorithms
"""
import numpy as np
from typing import List
from numpy.linalg import norm
from numpy import conjugate
from scipy import linalg
from unisys.basic import gate, Circuit
from unisys.utils.operator import params_u3


def arbitrary_2_qubit_state_circuit(state: np.ndarray, tqs: List[int] = None, return_u3: bool = True) -> Circuit:
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
    tqs = tqs if tqs else [0, 1]
    a0, a1, a2, a3 = state
    v1, v2 = state[:2], state[2:]
    if np.allclose(v1 @ v2, 0):

        k = np.linalg.norm(v2) / np.linalg.norm(v1)
    else:
        k = - np.linalg.norm(v2) / np.linalg.norm(v1) * (conjugate(v1) @ v2) / np.linalg.norm(conjugate(v1) @ v2)

    W1 = parametrized_su2(a3 - k * a1, conjugate(a2 - k * a0)).T

    psi_1 = linalg.block_diag(np.identity(2), gate.Z.data) @ np.kron(np.identity(2), W1) @ state

    b0, b1, b2, b3 = psi_1
    W2 = parametrized_su2(conjugate(b1), conjugate(b3))

    psi_2 = np.kron(W2, np.identity(2)) @ psi_1

    c0, c1, c2, c3 = psi_2
    W3 = parametrized_su2(conjugate(c0), -conjugate(c1)).T

    if return_u3:
        return Circuit([
            gate.U3(*params_u3(W2.conj().T)).on(tqs[0]),
            gate.U3(*params_u3(W3.conj().T)).on(tqs[1]),
            gate.Z.on(tqs[1], tqs[0]),
            gate.U3(*params_u3(W1.conj().T)).on(tqs[1]),
        ])
    else:
        return Circuit([
            gate.UnivGate(W2.conj().T, 'W2†').on(tqs[0]),
            gate.UnivGate(W3.conj().T, 'W3†').on(tqs[1]),
            gate.Z.on(tqs[1], tqs[0]),
            gate.UnivGate(W1.conj().T, 'W1†').on(tqs[1]),
        ])


def arbitrary_3_qubit_state_circuit(state: np.ndarray, tqs: List[int] = None, return_u3: bool = True) -> Circuit:
    if not np.allclose(norm(state), 1) or state.shape != (8,):
        raise ValueError('The input is not a reasonable 3-qubit state!')
    tqs = tqs if tqs else [0, 1, 2]
    raise NotImplementedError


def parametrized_su2(x, y) -> np.ndarray:
    r"""
    Return such a parametrized unitary operator belonging to SU(2)

    .. math::
        U\left( x,y \right) =\frac{1}{\sqrt{|x|^2+|y|^2}}\left( \begin{matrix}
            x   &		y\\
            -y^*    &		x^*\\
        \end{matrix} \right)
    """
    return np.array([
        [x, y],
        [-conjugate(y), conjugate(x)]
    ]) / np.linalg.norm([x, y])
