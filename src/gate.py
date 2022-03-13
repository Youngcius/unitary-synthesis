"""
Simple standard single-qubit gates and two-qubit gates
"""
import numpy as np
from scipy import linalg
from numpy import exp, sin, cos, sqrt
from copy import deepcopy


class Gate:
    """
    Single-qubit gate and controlled two-qubit gate class
    """

    def __init__(self, data: np.ndarray, tq=0, cq=None, name: str = None, *args, **kwargs):
        """
        :param data: unitary matrix data, ndarray type
        :param tq: target qubit index
        :param cq: control qubit index, optional for single-qubit gate
        :param name: name of the quantum gate
        """
        self.tq = tq
        self.cq = cq
        # if it is a two/multi-qubit gate, the first index is cq, later index/indices is/are tq
        self.qregs = [self.tq] if self.cq is None else [self.cq, self.tq]
        self.data = data.astype(complex)
        self.name = name
        if 'angle' in kwargs.keys():
            self.angle = kwargs['angle']
        if 'angles' in kwargs.keys():
            self.angles = kwargs['angles']

    def inverse(self):
        """
        Inverse, i.e. conjugate transpose, of the origin gate, only
        :return: a new Gate instance
        """
        if self.cq is None:
            t_names = ('T', 'TDG')
            s_names = ('S', 'SDG')
            g = deepcopy(self)
            g.data = self.data.conj().T
            if g.name in t_names:
                idx = t_names.index(g.name)
                g.name = t_names[(idx + 1) % 2]
            elif g.name in s_names:
                idx = s_names.index(g.name)
                g.name = s_names[(idx + 1) % 2]
            return g
        else:
            raise NotImplementedError('not implementted for controlled gate')

    def perm(self):
        """
        Permutation of tq and cq indices, only support fixed controlled gate
        :return: a new Gate instance
        """
        if self.cq is None:
            raise ValueError('only support controlled gate')
        else:
            # e.g. CX, XZ, CH
            cu = deepcopy(self)
            cu.tq, cu.cq = self.cq, self.tq
            cu.qregs = [cu.cq, cu.tq]
            if cu.name == 'CX':
                if cu.cq == 0:
                    cu.data = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                        [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                                        [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
                                        [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j]])
                else:
                    cu.data = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                        [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
                                        [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                                        [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j]])
            elif cu.name == 'CH':
                if cu.cq == 0:
                    cu.data = (np.kron(np.diag([1, 0]), np.identity(2)) + np.kron(np.diag([0, 1]), H.data)).astype(
                        complex)
                else:
                    cu.data = (np.kron(np.identity(2), np.diag([1, 0])) + np.kron(H.data, np.diag([0, 1]))).astype(
                        complex)
            else:
                raise ValueError('unsupported controlled gate currently')

        return cu

    def set_qregs(self, tq, cq=None):
        """
        For single-qubit gate and controlled-rotation gate
        :param tq: target qubit
        :param cq: control qubit
        :return: a new Gate instance
        """
        u = deepcopy(self)
        u.tq = tq
        if cq is not None:
            u.cq = cq
        u.qregs = [u.tq] if u.cq is None else [u.cq, u.tq]  # cq在前，tq在后
        return u


# Fixed gates: Executable quantum gates without parameters (1-qubit & 2-qubit gates)
fixed_gates = ['x', 'y', 'z', 'i', 'h', 's', 't', 'sdg', 'tdg', 'cx', 'cz', 'swap', 'ch']

I = Gate(np.identity(2), name='I')
X = Gate(np.array([[0. + 0.j, 1. + 0.j],
                   [1. + 0.j, 0. + 0.j]]), name='X')
Y = Gate(np.array([[0. + 0.j, 0. - 1.j],
                   [0. + 1.j, 0. + 0.j]]), name='Y')
Z = Gate(np.array([[1. + 0.j, 0. + 0.j],
                   [0. + 0.j, -1. + 0.j]]), name='Z')
H = Gate(np.array([[1, 1], [1, -1]]) / sqrt(2), name='H')
S = Gate(np.diag([1, 1j]), name='S')
T = Gate(np.diag([1, exp(1j * np.pi / 4)]), name='T')
CX = Gate(np.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 1., 0.]]), tq=1, cq=0, name='CX')
CZ = Gate(np.diag([1, 1, 1, -1]), name='CZ')
Swap = Gate(np.array([[1., 0., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.]]), tq=1, cq=0, name='Swap')
CH = Gate(np.kron(np.diag([1, 0]), np.identity(2)) + np.kron(np.diag([0, 1]), H.data), tq=1, cq=0, name='CH')

# Rotation gates: Executable quantum gates with parameters (1-qubit & 2-qubit gates)
rotation_gates = ['rx', 'ry', 'rz', 'crx', 'cry', 'crz', 'u3', 'cu3']


def Rx(theta: float, tq=0) -> Gate:
    data = linalg.expm(-1j * theta / 2 * X.data)
    return Gate(data, tq, name='Rx', angle=theta)


def Ry(theta: float, tq=0) -> Gate:
    data = linalg.expm(-1j * theta / 2 * Y.data)
    return Gate(data, tq, name='Ry', angle=theta)


def Rz(theta: float, tq=0) -> Gate:
    data = linalg.expm(-1j * theta / 2 * Z.data)
    return Gate(data, tq, name='Rz', angle=theta)


def U3(theta, phi, lamda, tq=0) -> Gate:
    """
    :math: `U3(\theta, \phi, \lambda) = exp(i(\phi + \theta)/2) RZ(\phi) RZ(\theta) RZ(\lambda)`
    """
    data = np.array([[cos(theta / 2), -exp(1j * lamda) * sin(theta / 2)],
                     [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lamda)) * cos(theta / 2)]])
    return Gate(data, tq, name='U3', angles=(theta, phi, lamda))


def CU3(theta, phi, lamda, tq=1, cq=0):
    data = np.array([[cos(theta / 2), -exp(1j * lamda) * sin(theta / 2)],
                     [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lamda)) * cos(theta / 2)]])
    return Gate(data, tq, cq, name='CU3', angles=(theta, phi, lamda))


def CRx(theta: float, tq=1, cq=0) -> Gate:
    data = linalg.expm(-1j * theta / 2 * X.data)
    return Gate(data, tq, cq, name='Rx', angle=theta)


def CRy(theta: float, tq=1, cq=0) -> Gate:
    data = linalg.expm(-1j * theta / 2 * Y.data)
    return Gate(data, tq, cq, name='Ry', angle=theta)


def CRz(theta: float, tq=1, cq=0) -> Gate:
    data = linalg.expm(-1j * theta / 2 * Z.data)
    return Gate(data, tq, cq, name='Rz', angle=theta)
