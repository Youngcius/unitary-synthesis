"""
Functions for Gate decomposition and matrix decomposition
"""
import numpy as np

import gate
from utils import *
from gate import Gate
from typing import List
from math import sqrt, pi, atan2
from circuit import optimize_circuit

M = np.array([[1, 0, 0, 1j],
              [0, 1j, 1, 0],
              [0, 1j, -1, 0],
              [1, 0, 0, -1j]]) / sqrt(2)

Mdag = M.conj().T

A = np.array([[1, 1, -1, 1],
              [1, 1, 1, -1],
              [1, -1, -1, -1],
              [1, -1, 1, 1]])


def params_zyz(U: np.ndarray):
    """
    ZYZ decomposition of 2*2 unitary operator
    :math:`U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)`
    :return: a series of phase angles
    """
    if U.shape != (2, 2):
        raise ValueError('U should be a 2*2 matrix')
    coe = linalg.det(U) ** (-0.5)
    alpha = - np.angle(coe)
    V = coe * U
    V = V.round(10)
    theta = 2 * atan2(abs(V[1, 0]), abs(V[0, 0]))
    phi_lam_sum = 2 * np.angle(V[1, 1])
    phi_lam_diff = 2 * np.angle(V[1, 0])
    phi = (phi_lam_sum + phi_lam_diff) / 2
    lam = (phi_lam_sum - phi_lam_diff) / 2
    return alpha, theta, phi, lam


def params_u3(U: np.ndarray, return_phase=False):
    """
    Obtain the global phase "p" appended to the standard U3 operator
    :math:`U = exp(i p) U3(\theta, \phi, \lambda)`
    """
    alpha, theta, phi, lam = params_zyz(U)
    phase = alpha - (phi + lam) / 2
    if return_phase:
        return phase, theta, phi, lam
    else:
        return theta, phi, lam


def params_abc(U: np.ndarray):
    """
    ABC decomposition of 2*2 unitary operator
    :return: a series
    """
    if U.shape != (2, 2):
        raise ValueError('U should be a 2*2 matrix')
    alpha, theta, phi, lam = params_zyz(U)
    A = gate.Rz(phi).data @ gate.Ry(theta / 2).data
    B = gate.Ry(-theta / 2).data @ gate.Rz(-(phi + lam) / 2).data
    C = gate.Rz((lam - phi) / 2).data
    return alpha, A, B, C


def abc_decomp(U: np.ndarray, cq=0, return_u3=False) -> List[Gate]:
    """
    Decompose two-qubit controlled gate based on ABC decomposition
    """
    tq = 1 - cq
    if cq == 0:
        alpha, theta, phi, lam = params_zyz(U[2:, 2:])
    else:
        alpha, theta, phi, lam = params_zyz(U[:2, :2])
    # ===========
    # C_list = [gate.Rz((lam - phi) / 2, tq)]
    # B_list = [gate.Rz(-(phi + lam) / 2, tq), gate.Ry(-theta / 2, tq)]
    # A_list = [gate.Ry(theta / 2, tq), gate.Rz(phi, tq)]
    # circuit = []
    # circuit.extend(C_list)
    # circuit.append(gate.CX if cq == 0 else gate.CX.perm())
    # circuit.extend(B_list)
    # circuit.append(gate.CX if cq == 0 else gate.CX.perm())
    # circuit.extend(A_list)
    # phase_gate = Gate(np.diag([1, np.exp(1j * alpha)]), cq)
    # circuit.append(phase_gate)
    # ===========
    if np.allclose(theta, 0):
        # U is a CRz gate
        # print(alpha, theta, phi, lam)
        Rx = np.exp(1j * alpha) * gate.Rx(phi + lam).data
        if is_equiv_unitary(Rx, gate.X.data):
            # just need one CNOT
            coe = times_two_matrix(Rx, gate.X.data)
            angle = np.angle(coe)
            return optimize_circuit([
                gate.H.set_qregs(tq),
                gate.CX if cq == 0 else gate.CX.perm(),
                gate.H.set_qregs(tq),
                gate.Rz(angle, cq)
            ])

    A = gate.Rz(phi).data @ gate.Ry(theta / 2).data
    B = gate.Ry(-theta / 2).data @ gate.Rz(-(phi + lam) / 2).data
    C = gate.Rz((lam - phi) / 2).data
    if return_u3:
        # regardless of global phases
        return optimize_circuit([
            gate.Rz((lam - phi) / 2, tq),  # C
            gate.CX if cq == 0 else gate.CX.perm(),
            gate.U3(*params_u3(B), tq),  # B
            gate.CX if cq == 0 else gate.CX.perm(),
            gate.U3(*params_u3(A), tq),  # A
            gate.Rz(alpha, cq)
        ])
    else:
        return optimize_circuit([
            Gate(C, tq),
            gate.CX if cq == 0 else gate.CX.perm(),
            Gate(B, tq),
            gate.CX if cq == 0 else gate.CX.perm(),
            Gate(A, tq),
            Gate(np.diag([1, np.exp(1j * alpha)]), cq)
        ])


def kron_decomp(M: np.ndarray):
    """
    Kronecker product decomposition (KPD) algorithm (4*4 matrix)
    Note: This function is not robust (without tolerance).
    """
    M00, M01, M10, M11 = M[:2, :2], M[:2, 2:], M[2:, :2], M[2:, 2:]
    K = np.vstack([M00.ravel(), M01.ravel(), M10.ravel(), M11.ravel()])
    # print(K)
    if np.linalg.matrix_rank(K) == 1:
        # form of tensor product
        l = [not np.allclose(np.zeros(4), K[i]) for i in range(4)]
        # print(l)
        idx = l.index(True)  # the first non-zero block
        B = K[idx]
        A = np.array([])
        for i in range(4):
            if l[i]:
                a = times_two_matrix(K[i], B)
            else:
                a = 0
            A = np.append(A, a)
        A = A.reshape(2, 2)
        B = B.reshape(2, 2)
        return A, B
    else:
        return None, None


def is_tensor_prod(U: np.ndarray):
    """
    Distinguish whether a 4*4 matrix is the tensor product of two 2*2 matrices
    """
    _, _ = kron_decomp(U)
    if _ is None:
        return False
    else:
        return True


def tensor_prod_decomp(U: np.ndarray, return_u3=False) -> List[Gate]:
    """
    Tensor product decomposition based on Strict KPD algorithm
    """
    A, B = kron_decomp(U)

    if return_u3:
        # regardless global phases
        circuit = [
            gate.U3(*params_u3(A), tq=0),
            gate.U3(*params_u3(B), tq=1),
        ]
    else:
        circuit = [Gate(A, tq=0), Gate(B, tq=1)]
    return optimize_circuit(circuit)


def simult_svd(A: np.ndarray, B: np.ndarray):
    """
    Simultaneous SVD of two matrices, based on Eckart-Young theorem
    :math:`A=U D_1 V^{\dagger}, B=U D_2 V^{\dagger}`
    :param A: real matrix
    :param B: real matrix
    :return: U, V, D1, D2
            U and V are both in SO(2)
    """
    if A.shape != B.shape:
        raise ValueError('A and B should have the same dimension')
    d = A.shape[0]
    # real orthogonal matrices
    Ua, Da, Vah = linalg.svd(A)
    # ensure det(Ua) == det(Va) == +1
    # if linalg.det(Ua) < 0:
    #     Ua[:, 0] *= -1
    # if linalg.det(Vah) < 0:
    #     Vah[0, :] *= -1
    Uah = Ua.conj().T
    Va = Vah.conj().T
    # Da = Uah @ A @ Va
    if np.count_nonzero(Da) == d:
        D = Da
        # G commutes with D
        G = Uah @ B @ Va
        # because G is hermitian, eig-decomposition is spectral decomposition
        Dg, P = linalg.eig(G)  # P 是 幺正 矩阵，当G是real的，P是实正交阵
        # ensure det(P) == det(Ua) == det(Va) == +1
        # if linalg.det(P) < 0:
        #     P[:, 0] *= -1
        U = Ua @ P
        V = Va @ P

        # D1 = P.conj().T @ np.diag(D) @ P
        # D2 = np.diag(Dg)
        if linalg.det(U) < 0:
            U[:, 0] *= -1
        if linalg.det(V) < 0:
            V[:, 0] *= -1

        D1 = U.conj().T @ A @ V
        D2 = U.conj().T @ B @ V
        return U, V, D1, D2
    else:
        # D =
        # return U, V,Da,Db
        # TODO: complete this block if necessary
        raise NotImplementedError('Not implemented yet for the situation that A is not full-rank')


def is_so4(U: np.ndarray):
    """
    Distinguish if one matrix is in SO(4) (4-dimension Special Orthogonal group)
    """
    if U.shape != (4, 4):
        raise ValueError('U should be a 4*4 matrix')
    return np.allclose(U @ U.conj().T, np.identity(4)) and np.allclose(linalg.det(U), 1)


def so4_to_magic_su2s(U: np.ndarray):
    """
    Decompose 1 SO(4) operator into 2 SU(2) operators with Magic matrix transformation: U = Mdag @ kron(A, B) @ M
    :param U: matrix of SO(4)
    :return: two SU(2) matrices
    """
    if is_so4(U):
        # KPD is definitely feasible when the input matrix is in SO(4)
        return kron_decomp(M @ U @ Mdag)
    else:
        raise ValueError('Input matrix is not in SO(4)')


def cnot_decomp(U: np.ndarray, return_u3=False) -> List[Gate]:
    """
    KAK decomposition (CNOT basis) of two-qubit gate.
    ---
    Step 1: decompose into
        :math:`\left( A_0 \otimes A_1 \right) e^{-iH}\left( B_0 \otimes B_1 \right) `
        ---B0------------------A0---
                | exp(-iH) |
        ---B1------------------A1---
    Step 2: calculate parameterized gates exp(-iH) with three CNOT gates
        ---B0---@---U0---@---V0---@---W--------A0---
                |        |        |
        ---B1---X---U1---X---V1---X---W^dag----A1---
    """

    I = gate.I.data
    X = gate.X.data
    Y = gate.Y.data
    Z = gate.Z.data
    Mdag = M.conj().T
    phase = glob_phase(U)
    # construct a new matrix replacing U
    UU = Mdag @ remove_glob_phase(U) @ M  # ensure the decomposed object is in SU(4)
    Ur = np.real(UU)  # real part of UU
    Ui = np.imag(UU)  # imagine part of UU

    # simultaneous SVD decomposition
    Ql, Qr, Dr, Di = simult_svd(Ur, Ui)
    D = Dr + 1j * Di

    # A0, A1 = so4_to_magic_su2s(Ql)
    # B0, B1 = so4_to_magic_su2s(Qr.T)
    _, A0, A1 = kron_factor_4x4_to_2x2s(M @ Ql @ Mdag)
    _, B0, B1 = kron_factor_4x4_to_2x2s(M @ Qr.T @ Mdag)

    k = linalg.inv(A) @ np.angle(np.diag(D))
    h1, h2, h3 = -k[1:]

    U0 = 1j / sqrt(2) * (X + Z) @ linalg.expm(-1j * (h1 - pi / 4) * X)
    V0 = -1j / sqrt(2) * (X + Z)
    U1 = linalg.expm(-1j * h3 * Z)
    V1 = linalg.expm(1j * h2 * Z)
    W = (I - 1j * X) / sqrt(2)

    # list of operators
    Ras = [B0, U0, V0, A0 @ W]
    Rbs = [B1, U1, V1, A1 @ W.conj().T]

    if return_u3:
        circuit = [
            gate.U3(*params_u3(Ras[0]), tq=0), gate.U3(*params_u3(Rbs[0]), tq=1),
            gate.CX,
            gate.U3(*params_u3(Ras[1]), tq=0), gate.U3(*params_u3(Rbs[1]), tq=1),
            gate.CX,
            gate.U3(*params_u3(Ras[2]), tq=0), gate.U3(*params_u3(Rbs[2]), tq=1),
            gate.CX,
            gate.U3(*params_u3(Ras[3]), tq=0), gate.U3(*params_u3(Rbs[3]), tq=1)
        ]
    else:
        circuit = [
            Gate(Ras[0], tq=0), Gate(Rbs[0], tq=1),
            gate.CX,
            Gate(Ras[1], tq=0), Gate(Rbs[1], tq=1),
            gate.CX,
            Gate(Ras[2], tq=0), Gate(Rbs[2], tq=1),
            gate.CX,
            Gate(Ras[3], tq=0), Gate(Rbs[3], tq=1)
        ]
    return optimize_circuit(circuit)


def decomp_gate(U: np.ndarray, return_u3: bool = False) -> List[Gate]:
    """
    High-level two-qubit gate decomposition function.
    As for the sequence of decomposed gates, they are executed from left to right on hardware.
    :param U: the two-qubit gate in U(4)
    :param return_u3:
                if True, each Gate instance of the returned circuit is a U3 gate, regardless of global phase
                if False, only return matrices with acted qubit indices, operator of the circuit is equal to U
    :return: a list of Gate instances
    """
    if U.shape != (4, 4):
        raise ValueError('U should be a 4*4 matrix')
    if not np.allclose(U @ U.T.conjugate(), np.identity(4)):
        raise ValueError('U should be a unitary matrix')

    if is_tensor_prod(U):
        # simple tensor product decomposition
        return tensor_prod_decomp(U, return_u3)
    elif is_equiv_unitary(U, gate.Swap.data):
        # 3 CNOT gates construct a Swap gate
        return [gate.CX, gate.CX.perm(), gate.CX]
    elif is_control_unitary(U) is not None:
        # ABC decomposition
        return abc_decomp(U, cq=is_control_unitary(U), return_u3=return_u3)
    else:
        # universal two-qubit gate decomposition --> SU(2) and 3 CNOT gates
        return cnot_decomp(U, return_u3)


def kron_factor_4x4_to_2x2s(mat: np.ndarray):
    """
    Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.
    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.
    Giving an incorrect matrix will cause garbage output.
    :param mat: 4x4 unitary matrix to factor.
    :return: A scalar factor and a pair of 2x2 unit-determinant matrices.
            The kronecker product of all three is equal to the given matrix.
    """
    # Use the entry with the largest magnitude as a reference point
    a, b = max(((i, j) for i in range(4)
                for j in range(4)), key=lambda t: abs(mat[t]))

    # Extract sub-factors touching the reference cell
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = mat[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = mat[a ^ i, b ^ j]

    # Rescale factors to have unit determinants
    f1 /= np.sqrt(np.linalg.det(f1)) or 1
    f2 /= np.sqrt(np.linalg.det(f2)) or 1

    # Determine global phase
    g = mat[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    return g, f1, f2
