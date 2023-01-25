"""
Simple standard single-qubit gates and two-qubit gates
"""
from typing import List, Union
from math import pi
import numpy as np
from scipy import linalg
from numpy import exp, sin, cos, sqrt
from copy import deepcopy
from ..utils.fucntions import is_power_of_two


class Gate:
    """Single-qubit gate and controlled two-qubit gate class"""

    def __init__(self, data: np.ndarray, name: str = None, *args, **kwargs):
        """

        Args:
            data: unitary matrix operating on target qubits, ndarray type
            tqs: target qubit indices
            cqs: control qubit indices, optional
            name: name of the quantum gate, optional
            *args:
            **kwargs:
        """
        self.name = name
        self._targ_qubits = []
        self._ctrl_qubits = []
        if not is_power_of_two(data.shape[0]) or not np.allclose(data @ data.conj().T, np.identity(data.shape[0])):
            raise ValueError('data is not valid quantum gate operator')
        self.n_qubits = int(np.log2(data.shape[0]))  # initial n_qubits is defined by data.shape
        self.data = data.astype(complex)

        # parameters for transforming to QSAM
        kwargs.setdefault('angle', None)
        kwargs.setdefault('angles', None)
        self.angle = kwargs['angle']
        self.angles = kwargs['angles']

    def on(self, tqs: Union[List[int], int], cqs: Union[List[int], int] = None):
        """
        Operate on specific qubits.

        Args:
            tqs: target qubit indices
            cqs: control qubit indices, optional

        Returns: Quantum Gate with operated qubit indices.
        """
        g = deepcopy(self)
        tqs = [tqs] if isinstance(tqs, int) else tqs
        cqs = [cqs] if isinstance(cqs, int) else cqs
        if isinstance(g, UnivGate) and len(tqs) != g.n_qubits:
            raise ValueError('number of target qubits does not coincide with the gate definition')

        if g.n_qubits > 1:  # only single-qubit gate can be expanded to identical tensor-product gate
            if len(tqs) > g.n_qubits or len(tqs) < g.n_qubits:
                raise ValueError('number of target qubits does not coincide with the gate definition')

        g.n_qubits = len(tqs)  # new number of target qubits
        g._targ_qubits = tqs
        if cqs:
            g._ctrl_qubits = cqs
        return g

    def __call__(self, *args, **kwargs):
        return self.on(*args, **kwargs)
    def __repr__(self):
        if not self._ctrl_qubits:
            return '{}: targ {}'.format(self.name, self._targ_qubits)
        return '{}: targ {} | ctrl {}'.format(self.name, self._targ_qubits, self._ctrl_qubits)
    # def perm(self):
    #     # TODO: delete this
    #     """
    #     Permutation of tq and cq indices, only support fixed controlled gate
    #     :return: a new Gate instance
    #     """
    #     if self.cq is None:
    #         raise ValueError('only support controlled gate')
    #     else:
    #         # e.g. CX, XZ, CH
    #         cu = deepcopy(self)
    #         cu.tq, cu.cq = self.cq, self.tq
    #         cu.qregs = [cu.cq, cu.tq]
    #         if cu.name == 'CX':
    #             if cu.cq == 0:
    #                 cu.data = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
    #                                     [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
    #                                     [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
    #                                     [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j]])
    #             else:
    #                 cu.data = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
    #                                     [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
    #                                     [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
    #                                     [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j]])
    #         elif cu.name == 'CH':
    #             if cu.cq == 0:
    #                 cu.data = (np.kron(np.diag([1, 0]), np.identity(2)) + np.kron(np.diag([0, 1]), H.data)).astype(
    #                     complex)
    #             else:
    #                 cu.data = (np.kron(np.identity(2), np.diag([1, 0])) + np.kron(H.data, np.diag([0, 1]))).astype(
    #                     complex)
    #         else:
    #             raise ValueError('unsupported controlled gate currently')
    #
    #     return cu

    #
    # def set_qregs(self, tq, cq=None):
    #     """
    #     For single-qubit gate and controlled-rotation gate
    #     :param tq: target qubit
    #     :param cq: control qubit
    #     :return: a new Gate instance
    #     """
    #     u = deepcopy(self)
    #     u.tq = tq
    #     if cq is not None:
    #         u.cq = cq
    #     u.qregs = [u.tq] if u.cq is None else [u.cq, u.tq]  # cq在前，tq在后
    #     return u

    @property
    def tq(self):
        if len(self._targ_qubits) > 1:
            raise ValueError('Gate {} has more than 1 target qubit'.format(self.name))
        if not self._targ_qubits:
            raise ValueError('Gate {} has no target qubit'.format(self.name))
        return self._targ_qubits[0]

    @property
    def cq(self):
        if len(self._ctrl_qubits) > 1:
            raise ValueError('Gate {} has more than 1 control qubit'.format(self.name))
        if not self._ctrl_qubits:
            raise ValueError('Gate {} has no control qubit'.format(self.name))
        return self._ctrl_qubits[0]

    @property
    def tqs(self):
        return self._targ_qubits

    @property
    def cqs(self):
        return self._ctrl_qubits

    @property
    def qregs(self):
        return self._targ_qubits if not self._ctrl_qubits else self._ctrl_qubits + self._targ_qubits

    def hermitian(self):
        """
        Hermitian conjugate of the origin gate
        Returns: a new Gate instance
        """
        g = deepcopy(self)
        g.data = g.data.conj().T
        if g.name.endswith('†'):
            g.name = g.name[:-1]
        else:
            g.name = g.name + '†'
        return g


class UnivGate(Gate):
    """
    Universal quantum gate
    """

    def __init__(self, data, name=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.name = name if name else 'U'


class XGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[0. + 0.j, 1. + 0.j],
                                   [1. + 0.j, 0. + 0.j]]), name='X', *args, **kwargs)


class YGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[0. + 0.j, -1.j],
                                   [1.j, 0. + 0.j]]), name='Y', *args, **kwargs)


class ZGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, -1. + 0.j]]), name='Z', *args, **kwargs)


class IGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.identity(2).astype(complex), name='I', *args, **kwargs)


class SGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 1.j]]), name='S', *args, **kwargs)


class TGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, np.exp(1.j * pi / 4)]]), name='T', *args, **kwargs)


class HGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 1. + 0.j],
                                   [1. + 0.j, -1. + 0.j]]) / sqrt(2), name='H', *args, **kwargs)


class SWAPGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]), name='SWAP', *args, **kwargs)


class ISWAPGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 1.j, 0. + 0.j],
                                   [0. + 0.j, 1.j, 0. + 0.j, 0. + 0.j],
                                   [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]), name='ISWAP', *args, **kwargs)


# Fixed gates: Executable quantum gates without parameters (1-qubit & 2-qubit gates)
fixed_gates = ['x', 'y', 'z', 'i', 'h', 's', 't', 'sdg', 'tdg', 'cx', 'cz', 'swap', 'ch']  # TODO: modify this list


# Rotation Gate
class RX(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * X.data), name='RX', angle=theta, *args, **kwargs)


class RY(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * Y.data), name='RY', angle=theta, *args, **kwargs)


class RZ(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * Z.data), name='RZ', angle=theta, *args, **kwargs)


class PhaseShift(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(np.array([[1. + 0., 0. + 0.],
                                   [0. + 0., np.exp(1j * theta)]]), name='PhaseShift', angle=theta, *args, **kwargs)


class GlobalPhase(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(np.array([[np.exp(1j * theta), 0. + 0.],
                                   [0. + 0., np.exp(1j * theta)]]), name='GlobalPhase', angle=theta, *args, **kwargs)


class XPow(Gate):
    """X power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (X.data - I.data)), name='XPow', *args, **kwargs)
        assert self.data == np.exp(1j * exponent * pi / 2) * RX(pi * exponent).data


class YPow(Gate):
    """Y power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (Y.data - I.data)), name='YPow' * args, **kwargs)
        assert self.data == np.exp(1j * exponent * pi / 2) * RY(pi * exponent).data


class ZPow(Gate):
    """Z power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (Z.data - I.data)), name='ZPow', *args, **kwargs)
        assert self.data == np.exp(1j * exponent * pi / 2) * RZ(pi * exponent).data


class U3(Gate):
    """
    U3 gate
    """

    def __init__(self, theta, phi, lamda, *args, **kwargs):
        super().__init__(np.array([[cos(theta / 2), -exp(1j * lamda) * sin(theta / 2)],
                                   [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lamda)) * cos(theta / 2)]]),
                         name='U3', angles=(theta, phi, lamda), *args, **kwargs)


# Rotation gates: Executable quantum gates with parameters (1-qubit & 2-qubit gates)
# rotation_gates = ['rx', 'ry', 'rz', 'crx', 'cry', 'crz', 'u3', 'cu3']
rotation_gates = ['rx', 'ry', 'rz', 'u3']  # TODO: modify QSAM parser


# Non-operation Gate

class Barrier:
    # TODO
    def __init__(self):
        ...


class Measurement:
    # TODO
    def __init__(self):
        ...


#
# def Rx(theta: float, tq=0) -> Gate:
#     data = linalg.expm(-1j * theta / 2 * X.data)
#     return Gate(data, tq, name='Rx', angle=theta)
#
#
# def Ry(theta: float, tq=0) -> Gate:
#     data = linalg.expm(-1j * theta / 2 * Y.data)
#     return Gate(data, tq, name='Ry', angle=theta)
#
#
# def Rz(theta: float, tq=0) -> Gate:
#     data = linalg.expm(-1j * theta / 2 * Z.data)
# #     return Gate(data, tq, name='Rz', angle=theta)
#
#
# def U3(theta, phi, lamda, tq=0) -> Gate:
#     """
#     :math: `U3(\theta, \phi, \lambda) = exp(i(\phi + \theta)/2) RZ(\phi) RZ(\theta) RZ(\lambda)`
#     """
#     data = np.array([[cos(theta / 2), -exp(1j * lamda) * sin(theta / 2)],
#                      [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lamda)) * cos(theta / 2)]])
#     return Gate(data, tq, name='U3', angles=(theta, phi, lamda))
#
#
# def CRx(theta: float, tq=1, cq=0) -> Gate:
#     data = linalg.expm(-1j * theta / 2 * X.data)
#     return Gate(data, tq, cq, name='Rx', angle=theta)

#
# def CRy(theta: float, tq=1, cq=0) -> Gate:
#     data = linalg.expm(-1j * theta / 2 * Y.data)
#     return Gate(data, tq, cq, name='Ry', angle=theta)
#
#
# def CRz(theta: float, tq=1, cq=0) -> Gate:
#     data = linalg.expm(-1j * theta / 2 * Z.data)
#     return Gate(data, tq, cq, name='Rz', angle=theta)
#
#
# def CU3(theta, phi, lamda, tq=1, cq=0):
#     data = np.array([[cos(theta / 2), -exp(1j * lamda) * sin(theta / 2)],
#                      [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lamda)) * cos(theta / 2)]])
#     return Gate(data, tq, cq, name='CU3', angles=(theta, phi, lamda))
#

X = XGate()
Y = YGate()
Z = ZGate()
I = IGate()
S = SGate()
T = TGate()
H = HGate()
SWAP = SWAPGate()
ISWAP = ISWAPGate()
