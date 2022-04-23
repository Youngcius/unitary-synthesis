"""
Utils functions
"""
import numpy as np
from scipy import linalg
from functools import reduce


def glob_phase(U: np.ndarray) -> float:
    """
    Compute the global phase `\alpha` from a d*d matrix
    :math:`U = e^{i\alpha} S`, s.t. S in SU(d)
    :return: `\alpha` is in (-pi, pi]
    """
    d = U.shape[0]
    exp_alpha = linalg.det(U) ** (1 / d)
    alpha = np.angle(exp_alpha)
    return alpha


def remove_glob_phase(U: np.ndarray) -> np.ndarray:
    """
    Remove the global phase of a d*d unitary matrix
    :math:`U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)`
    That is, remove :math:`e^{i\alpha}`
    """
    alpha = glob_phase(U)
    return U * np.exp(- 1j * alpha)


def is_equiv_unitary(U: np.ndarray, V: np.ndarray):
    """
    Regardless of the global phase, this function distinguishes whether two unitary operator is equivalent
    (considering some precision)
    """
    Uf = U.ravel()
    Vf = V.ravel()
    idx_Uf = np.flatnonzero(Uf.round(6))
    idx_Vf = np.flatnonzero(Vf.round(6))
    try:
        if np.allclose(idx_Uf, idx_Vf):
            coes = Uf[idx_Uf] / Vf[idx_Vf]
            return np.allclose(coes / coes[0], np.ones(len(idx_Uf)))
        else:
            return False
    except:
        return False


def is_control_unitary(U: np.ndarray):
    is_control_0 = is_equiv_unitary(U[:2, :2], np.identity(2)) and np.allclose(U[2:, 2:] @ U[2:, 2:].T.conjugate(),
                                                                               np.identity(2))
    is_control_1 = is_equiv_unitary(U[2:, 2:], np.identity(2)) and np.allclose(U[:2, :2] @ U[:2, :2].T.conjugate(),
                                                                               np.identity(2))
    if is_control_0:
        return 0
    elif is_control_1:
        return 1
    else:
        return None


def tensor_1_slot(U: np.ndarray, n: int, tq: int):
    """
    Given a single-qubit gate, compute its tensor unitary matrix expanded
    to the whole Hilbert space (totally n qubits).
    """
    if tq not in range(n):
        raise ValueError('the qubit idx is out of range')
    ops = [np.identity(2)] * n
    ops[tq] = U
    return reduce(np.kron, ops)


def tensor_2_slot(U: np.ndarray, n: int, cq: int, tq: int):
    """
    Given a two-qubit gate, compute the tensor unitary matrix expanded
    to the whole Hilbert space (totally n qubits) of a two-qubit gate.
    """
    if cq not in range(n) or tq not in range(n):
        raise ValueError('the qubit idx is out of range')
    arr_list = [np.identity(2)] * (n - 1)
    arr_list[0] = U
    res = reduce(np.kron, arr_list).reshape([2] * 2 * n)
    idx = np.repeat(-1, n)
    idx[cq] = 0
    idx[tq] = 1
    idx[idx < 0] = range(2, n)
    idx = idx.tolist()
    idx_latter = [i + n for i in idx]
    res = np.transpose(res, idx + idx_latter).reshape([2 ** n, 2 ** n])
    return res


def times_two_matrix(U: np.ndarray, V: np.ndarray):
    """
    Calculate the coefficient a, s.t. U = a V
    """
    assert U.shape == V.shape, "input matrices should have the same dimension"
    idx1 = np.flatnonzero(U.round(6))  # cut to some precision
    idx2 = np.flatnonzero(V.round(6))
    try:
        if np.allclose(idx1, idx2):
            return U.ravel()[idx1[0]] / V.ravel()[idx2[0]]
    except:
        return None
