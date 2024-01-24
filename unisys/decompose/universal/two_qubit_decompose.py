"""Two-qubit gate decomposition."""

from math import sqrt, pi
import numpy as np
from scipy import linalg

from unisys.basic import gate, Gate, Circuit
from unisys.basic.circuit import optimize_circuit
from unisys.decompose.fixed.pauli_related import crx_decompose, cry_decompose, crz_decompose
from unisys.utils.operations import M, M_DAG, A
from unisys.utils.operations import params_abc, params_zyz, params_u3
from unisys.utils.operations import kron_decomp, is_tensor_prod, kron_factor_4x4_to_2x2s
from unisys.utils.operations import remove_glob_phase, simult_svd


def tensor_product_decompose(g: Gate, return_u3: bool = True) -> Circuit:
    """
    Tensor product decomposition of a 2-qubit gate.

    Args:
        g (Gate): 2-qubit gate composed by tensor product
        return_u3 (bool): return gates in form of `U3` or `UnivMathGate`

    Returns:
        Circuit, including two single-qubit gates.
    """
    if len(g.tqs) != 2 or g.cqs:
        raise ValueError(f'{g} is not a 2-qubit gate with designated qubits')
    if not is_tensor_prod(g.data):
        raise ValueError(f'{g} is not a tensor-product unitary gate.')

    u0, u1 = kron_decomp(g.data)
    circ = Circuit()
    if return_u3:
        circ.append(gate.U3(*params_u3(u0)).on(g.tqs[0]))
        circ.append(gate.U3(*params_u3(u1)).on(g.tqs[1]))
    else:
        circ.append(gate.UnivGate(u0, 'U0').on(g.tqs[0]))
        circ.append(gate.UnivGate(u1, 'U1').on(g.tqs[1]))
    return optimize_circuit(circ)


def abc_decompose(g: Gate, return_u3: bool = True) -> Circuit:
    """
    Decompose two-qubit controlled gate via ABC decomposition.

    Args:
        g (QuantumGate): quantum gate with 1 control bit and 1 target bit.
        return_u3 (bool): return gates in form of `U3` or `UnivMathGate`

    Returns:
        Circuit, including at most 2 CNOT gates and 4 single-qubit gates.
    """

    if len(g.cqs) != 1 or len(g.tqs) != 1:
        raise ValueError(f'{g} is not a two-qubit controlled gate with designated qubits')
    if isinstance(g, gate.RX):
        return crx_decompose(g)
    if isinstance(g, gate.RY):
        return cry_decompose(g)
    if isinstance(g, gate.RZ):
        return crz_decompose(g)

    cq = g.cq
    tq = g.tq
    _, (_, phi, lam) = params_zyz(g.data)
    alpha, (a, b, c) = params_abc(g.data)
    circ = Circuit()
    if return_u3:
        # regardless global phases
        circ.append(
            gate.RZ((lam - phi) / 2).on(tq),
            gate.X.on(tq, cq),
            gate.U3(*params_u3(b)).on(tq),
            gate.X.on(tq, cq),
            gate.U3(*params_u3(a)).on(tq),
            gate.RZ(alpha).on(cq),
        )

    else:
        circ.append(
            gate.UnivGate(c, 'C').on(tq),
            gate.X.on(tq, cq),
            gate.UnivGate(b, 'B').on(tq),
            gate.X.on(tq, cq),
            gate.UnivGate(a, 'A').on(tq),
            gate.PhaseShift(alpha).on(cq)
        )

    return optimize_circuit(circ)


def kak_decompose(g: Gate, return_u3: bool = True) -> Circuit:
    r"""
    KAK decomposition (CNOT basis) of an arbitrary two-qubit gate.

    Step 1: decompose it into

             ┌──────────┐
        ──B0─┤          ├─A0──
             │ exp(-iH) │
        ──B1─┤          ├─A1──
             └──────────┘
    .. math::

        \left( A_0 \otimes A_1 \right) e^{-iH}\left( B_0 \otimes B_1 \right)

    Step 2: synthesize parameterized gates exp(-iH) using three CNOT gates

        ──B0────●────U0────●────V0────●────W─────A0──
                │          │          │
        ──B1────X────U1────X────V1────X────W†────A1──

    Args:
        g (QuantumGate): 2-qubit quantum gate
        return_u3 (bool): return gates in form of `U3` or `UnivMathGate`

    Returns:
        Circuit, including at most 3 CNOT gates and 6 single-qubit gates.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """
    if len(g.tqs) != 2 or g.cqs:
        raise ValueError(f'{g} is not an arbitrary 2-qubit gate with designated qubits')
    pauli_i = gate.I.data
    pauli_x = gate.X.data
    pauli_z = gate.Z.data

    # construct a new matrix replacing U
    u_su4 = M_DAG @ remove_glob_phase(g.data) @ M  # ensure the decomposed object is in SU(4)
    ur = np.real(u_su4)  # real part of u_su4
    ui = np.imag(u_su4)  # imagine part of u_su4

    # simultaneous SVD decomposition
    (q_left, q_right), (dr, di) = simult_svd(ur, ui)
    d = dr + 1j * di

    _, a0, a1 = kron_factor_4x4_to_2x2s(M @ q_left @ M_DAG)
    _, b0, b1 = kron_factor_4x4_to_2x2s(M @ q_right.T @ M_DAG)

    k = linalg.inv(A) @ np.angle(np.diag(d))
    h1, h2, h3 = -k[1:]

    u0 = 1j / sqrt(2) * (pauli_x + pauli_z) @ linalg.expm(-1j * (h1 - pi / 4) * pauli_x)
    v0 = -1j / sqrt(2) * (pauli_x + pauli_z)
    u1 = linalg.expm(-1j * h3 * pauli_z)
    v1 = linalg.expm(1j * h2 * pauli_z)
    w = (pauli_i - 1j * pauli_x) / sqrt(2)

    # list of operators
    rots1 = [b0, u0, v0, a0 @ w]  # rotation gate on idx1
    rots2 = [b1, u1, v1, a1 @ w.conj().T]

    idx1, idx2 = g.tqs
    circ = Circuit()
    if return_u3:
        circ.append(
            gate.U3(*params_u3(rots1[0])).on(idx1),
            gate.U3(*params_u3(rots2[0])).on(idx2),
            gate.X.on(idx2, idx1),
            gate.U3(*params_u3(rots1[1])).on(idx1),
            gate.U3(*params_u3(rots2[1])).on(idx2),
            gate.X.on(idx2, idx1),
            gate.U3(*params_u3(rots1[2])).on(idx1),
            gate.U3(*params_u3(rots2[2])).on(idx2),
            gate.X.on(idx2, idx1),
            gate.U3(*params_u3(rots1[3])).on(idx1),
            gate.U3(*params_u3(rots2[3])).on(idx2)
        )
    else:
        circ.append(
            gate.UnivGate(rots1[0], 'B0').on(idx1),
            gate.UnivGate(rots2[0], 'B1').on(idx2),
            gate.X.on(idx2, idx1),
            gate.UnivGate(rots1[1], 'U0').on(idx1),
            gate.UnivGate(rots2[1], 'U1').on(idx2),
            gate.X.on(idx2, idx1),
            gate.UnivGate(rots1[2], 'V0').on(idx1),
            gate.UnivGate(rots2[2], 'V1').on(idx2),
            gate.X.on(idx2, idx1),
            gate.UnivGate(rots1[3], 'W0').on(idx1),
            gate.UnivGate(rots2[3], 'W1').on(idx2)
        )

    return optimize_circuit(circ)


def can_decompose(g: Gate, return_u3: bool = True) -> Circuit:
    """
    Similar to KAK decomposition, but returning Canonical with single-qubit gates.
    """
    if len(g.tqs) != 2 or g.cqs:
        raise ValueError(f'{g} is not an arbitrary 2-qubit gate with designated qubits')

    # construct a new matrix replacing U
    u_su4 = M_DAG @ remove_glob_phase(g.data) @ M  # ensure the decomposed object is in SU(4)
    ur = np.real(u_su4)  # real part of u_su4
    ui = np.imag(u_su4)  # imagine part of u_su4

    # simultaneous SVD decomposition
    (q_left, q_right), (dr, di) = simult_svd(ur, ui)
    d = dr + 1j * di

    _, a0, a1 = kron_factor_4x4_to_2x2s(M @ q_left @ M_DAG)
    _, b0, b1 = kron_factor_4x4_to_2x2s(M @ q_right.T @ M_DAG)

    k = linalg.inv(A) @ np.angle(np.diag(d))
    # t1, t2, t3 = - k[1:] * 2 / np.pi
    theta1, theta2, theta3 = - k[1:] * 2
    circ = Circuit()
    if return_u3:
        circ.append(
            gate.U3(*params_u3(b0)).on(g.tqs[0]),
            gate.U3(*params_u3(b1)).on(g.tqs[1]),
            # gate.Can(t1, t2, t3).on(g.tqs),
            gate.Can(theta1, theta2, theta3).on(g.tqs),
            gate.U3(*params_u3(a0)).on(g.tqs[0]),
            gate.U3(*params_u3(a1)).on(g.tqs[1]),
        )
    else:
        circ.append(
            gate.UnivGate(b0, 'B0').on(g.tqs[0]),
            gate.UnivGate(b1, 'B1').on(g.tqs[1]),
            # gate.Can(t1, t2, t3).on(g.tqs),
            gate.Can(theta1, theta2, theta3).on(g.tqs),
            gate.UnivGate(a0, 'A0').on(g.tqs[0]),
            gate.UnivGate(a1, 'A1').on(g.tqs[1]),
        )

    return optimize_circuit(circ)
