"""
Triangular decomposition of a unitary matrix due to Reck et al.
"""
import numpy as np
from math import sin, cos
from copy import deepcopy
from functools import reduce


def Tmn(m: int, n: int, theta: float, phi: float, N: int):
    """
    Return a unitary matrix with non-trivial elements arrangement in form of :math: `\left( \begin{matrix}
                e^{i\phi}&		0\\
                0&		1\\
            \end{matrix} \right) \left( \begin{matrix}
                \sin \theta&		\cos \theta\\
                \cos \theta&		-\sin \theta\\
            \end{matrix} \right)`
    :param m: the first non-trivial index
    :param n: the second non-trivial index
    :param theta: rotation angle
    :param phi: phase angle
    :param N: dimension of the whole Hilbert space
    :return: N*N matrix, np.ndarray
    """
    T = np.identity(N).astype(complex)
    data = np.diag([np.exp(1j * phi), 1]) @ np.array(
        [
            [sin(theta), cos(theta)],
            [cos(theta), -sin(theta)]
        ], dtype=complex
    )
    T[m, m] = data[0, 0]
    T[n, n] = data[1, 1]
    T[m, n] = data[0, 1]
    T[n, m] = data[1, 0]
    return T


def reck_decomp(U: np.ndarray):
    """
    Reck decomposition of an arbitrary-dimension unitary matrix
    :param U: unitary matrix
    :return: a list of BS gate parameters, diagonal matrix
    """
    if U.shape[0] != U.shape[1]:
        raise ValueError('U is not a square matrix')
    N = U.shape[0]
    if not np.allclose(U.conj().T @ U, np.identity(N)):
        raise ValueError('U is not a unitary matrix')
    if not np.allclose(U, np.real(U)):
        raise ValueError('currently only supports real matrix')
    V = deepcopy(U)
    params_T = []
    for n in range(N - 1, 0, -1):
        n_vec = V[n][:(n + 1)]  # the last row, also the n-th row (0 is the initial index)
        # len(n_vec) == n + 1
        sign = 1
        theta_list = []
        params_T_tmp = []
        prod_list = []  # cos, sin

        for m in range(n):
            # m in [0, ..., n-1]
            prod = sign * n_vec[m]
            # phi_m = - np.angle(prod) TODO: remedy this to support complex matrix
            phi_m = 0
            prod = np.real(prod)
            if m == 0:
                theta_m = np.arccos(prod)
                theta_list.append(theta_m)
            elif m == n - 1:
                # determine the unique "theta" from "cos(theta)" and "sin(theta)" in range of [-pi, pi]
                sin_prod = np.prod([sin(theta) for theta in theta_list])
                cos_theta = prod / sin_prod
                sign = -sign
                prod_lat = np.real(sign * n_vec[n])
                sin_theta = prod_lat / sin_prod
                theta_m = cal_theta(sin_theta, cos_theta)
                theta_list.append(theta_m)
            else:
                sin_prod = np.prod([sin(theta) for theta in theta_list])
                theta_m = np.arccos(prod / sin_prod)
                theta_list.append(theta_m)

            prod_list.append(prod)
            params_T_tmp.append((m, n, theta_m, phi_m, N))
            sign = -sign

        params_T_tmp = list(reversed(params_T_tmp))
        Rtmp = reduce(np.dot, [Tmn(*param) for param in params_T_tmp])
        V = V @ Rtmp
        params_T.extend(params_T_tmp)
    return params_T, np.diag(V)


def cal_theta(s, c):
    """
    Calculate an angle value, s.t. sin(theta) == s and cos(theta) == c
    :param s: sin(theta)
    :param c: cos(theta)
    :return: theta, type: float, unit: rad, range: [-pi, pi]
    """
    if not np.allclose(s ** 2 + c ** 2, 1):
        raise ValueError('"s" and "c" are not a reasonable sine and cos values')
    arcs = np.arcsin(s)  # range: [-pi/2, pi/2]
    arcc = np.arccos(c)  # range: [0, pi]
    if not np.allclose(arcs, arcc):
        # remedy angle range
        if np.allclose(np.pi - arcs, arcc) and arcs >= 0 and arcs <= np.pi / 2:
            # 1) range: [pi/2, pi]
            return arcc
        elif np.allclose(arcs, -arcc) and arcs >= -np.pi / 2 and arcs <= 0:
            # 2) range: [-pi/2, 0]
            return arcs
        else:
            # 3) range: [-pi, -pi/2]
            assert np.allclose(-np.pi - arcs, -arcc), "?????????"
            return -np.pi - arcs
    else:
        # range: [0, pi/2]
        return arcs
