"""
Quantum Gate
"""
import numpy as np
from typing import List, Union
from math import pi
from scipy import linalg
from numpy import exp, sin, cos, sqrt
from copy import copy
from unisys.utils.functions import is_power_of_two


class Gate:
    """Single-qubit gate and controlled two-qubit gate class"""

    def __init__(self, data: np.ndarray, name: str = None, *args, **kwargs):
        """

        Args:
            data: unitary matrix operating on target qubits, ndarray type
            tqs: target qubit indices
            cqs: control qubit indices, optional
            name: name of the quantum gate, optional
        """
        self.name = name
        self._targ_qubits = []
        self._ctrl_qubits = []
        if not is_power_of_two(data.shape[0]) or not np.allclose(data @ data.conj().T, np.identity(data.shape[0])):
            raise ValueError('data is not valid quantum gate operator')
        self.n_qubits = int(np.log2(data.shape[0]))  # initial n_qubits is defined by data.shape
        self.data = data.astype(complex)

        # parameters for transforming to QASM
        self.angle = kwargs.get('angle', None)
        self.angles = kwargs.get('angles', None)
        self.params = kwargs.get('params', None)
        self.exponent = kwargs.get('exponent', None)

    def __hash__(self):
        return hash(id(self))

    def copy(self):
        return copy(self)

    def on(self, tqs: Union[List[int], int], cqs: Union[List[int], int] = None):
        """
        Operate on specific qubits.

        Args:
            tqs: target qubit indices
            cqs: control qubit indices, optional

        Returns: Quantum Gate with operated qubit indices.
        """
        g = self.copy()
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

    def __repr__(self) -> str:
        prefix = self.name
        if self.angle:
            prefix += '({:.2f}π)'.format(self.angle / pi)
        elif self.angles:
            prefix += '({})'.format(','.join(['{:.2f}π'.format(a / pi) for a in self.angles]))
        elif self.params:
            prefix += '({})'.format(','.join([str(p) for p in self.params]))
        elif self.exponent:
            prefix += '({})'.format(self.exponent)

        tqs_str = str(self._targ_qubits[0]) if len(self._targ_qubits) == 1 else '|'.join(
            [str(tq) for tq in self._targ_qubits])
        cqs_str = str(self._ctrl_qubits[0]) if len(self._ctrl_qubits) == 1 else '|'.join(
            [str(cq) for cq in self._ctrl_qubits])
        if not self._targ_qubits:
            return prefix
        if not self._ctrl_qubits:
            return '{}{{{}}}'.format(prefix, tqs_str)
        return '{}{{{}←{}}}'.format(prefix, tqs_str, cqs_str)

    def math_repr(self):
        tqs_str = str(self._targ_qubits[0]) if len(self._targ_qubits) == 1 else '|'.join(
            [str(tq) for tq in self._targ_qubits])
        cqs_str = str(self._ctrl_qubits[0]) if len(self._ctrl_qubits) == 1 else '|'.join(
            [str(cq) for cq in self._ctrl_qubits])
        g_name = self.name
        if g_name == 'SDG':
            g_name = 'S^\dagger'
        if g_name == 'TDG':
            g_name = 'T^\dagger'
        if not self._ctrl_qubits:
            return '${}_{{{}}}$'.format(g_name, tqs_str)
        return '${}_{{{}←{}}}$'.format(g_name, tqs_str, cqs_str)

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

    @property
    def num_qregs(self):
        """Number of qubits operated by the gate (both target and control qubits)"""
        return len(self.qregs)

    def hermitian(self):
        """
        Hermitian conjugate of the origin gate

        Returns: a new Gate instance
        """
        g = self.copy()
        g.data = g.data.conj().T

        t_names = ('T', 'TDG')
        s_names = ('S', 'SDG')
        if g.name in t_names:
            idx = t_names.index(g.name)
            g.name = t_names[(idx + 1) % 2]
        elif g.name in s_names:
            idx = s_names.index(g.name)
            g.name = s_names[(idx + 1) % 2]
        elif g.name.endswith('†'):
            g.name = g.name[:-1]
        elif g.name in ROTATION_GATES:
            raise NotImplementedError
        elif np.allclose(g.data, self.data):
            pass
        else:
            g.name = g.name + '†'

        return g


class UnivGate(Gate):
    """Universal quantum gate"""

    def __init__(self, data, name=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.name = name if name else 'U'


# Fixed Gate
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


class SDGGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, -1.j]]), name='SDG', *args, **kwargs)


class TGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, np.exp(1.j * pi / 4)]]), name='T', *args, **kwargs)


class TDGGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array([[1. + 0.j, 0. + 0.j],
                                   [0. + 0.j, np.exp(- 1.j * pi / 4)]]), name='TDG', *args, **kwargs)


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
                                   [0. + 0., np.exp(1j * theta)]]), name='P', angle=theta, *args, **kwargs)


class GlobalPhase(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(np.array([[np.exp(1j * theta), 0. + 0.],
                                   [0. + 0., np.exp(1j * theta)]]), name='GlobalPhase', angle=theta, *args, **kwargs)


class XPow(Gate):
    """X power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (X.data - I.data)), name='XPow', exponent=exponent,
                         *args, **kwargs)
        assert np.allclose(self.data, np.exp(1j * exponent * pi / 2) * RX(pi * exponent).data)


class YPow(Gate):
    """Y power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (Y.data - I.data)), name='YPow', exponent=exponent,
                         *args, **kwargs)
        assert np.allclose(self.data, np.exp(1j * exponent * pi / 2) * RY(pi * exponent).data)


class ZPow(Gate):
    """Z power gate"""

    def __init__(self, exponent, *args, **kwargs):
        super().__init__(linalg.expm(-1j * exponent * pi / 2 * (Z.data - I.data)), name='ZPow', exponent=exponent,
                         *args, **kwargs)
        assert np.allclose(self.data, np.exp(1j * exponent * pi / 2) * RZ(pi * exponent).data)


class U1(Gate):
    """U1 gate"""

    def __init__(self, lamda, *args, **kwargs):
        super().__init__(np.array([[1. + 0., 0. + 0.],
                                   [0. + 0., np.exp(1j * lamda)]]), name='U1', angle=lamda, *args, **kwargs)


class U2(Gate):
    """U2 gate"""

    def __init__(self, phi, lamda, *args, **kwargs):
        super().__init__(np.array([[1, - exp(1j * lamda)],
                                   [exp(1j * phi), exp(1j * (phi + lamda))]]) / sqrt(2),
                         name='U2', angles=(phi, lamda), *args, **kwargs)


class U3(Gate):
    """U3 gate"""

    def __init__(self, theta, phi, lamda, *args, **kwargs):
        super().__init__(np.array([[cos(theta / 2), -exp(1j * lamda) * sin(theta / 2)],
                                   [exp(1j * phi) * sin(theta / 2), exp(1j * (phi + lamda)) * cos(theta / 2)]]),
                         name='U3', angles=(theta, phi, lamda), *args, **kwargs)


class RXX(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * np.kron(X.data, X.data)), name='RXX', angle=theta, *args,
                         **kwargs)


class RYY(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * np.kron(Y.data, Y.data)), name='RYY', angle=theta, *args,
                         **kwargs)


class RZZ(Gate):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(linalg.expm(-1j * theta / 2 * np.kron(Z.data, Z.data)), name='RZZ', angle=theta, *args,
                         **kwargs)

def exp_xx(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(X.data, X.data))
def exp_yy(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(Y.data, Y.data))
def exp_zz(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(Z.data, Z.data))
def exp_xy(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(X.data, Y.data))
def exp_yx(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(Y.data, X.data))
def exp_xz(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(X.data, Z.data))
def exp_zx(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(Z.data, X.data))
def exp_yz(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(Y.data, Z.data))
def exp_zy(theta):
    return linalg.expm(-1j * theta / 2 * np.kron(Z.data, Y.data))


class WeylGate(Gate):
    r"""
    Canonical gate with respect to Weyl chamber

    .. math::
        \textrm{Can}(\theta_1, \theta_2, \theta_3) = e^{- i \frac{1}{2}(\theta_1 XX + \theta_2 YY + \theta_3 ZZ)}

    Or

    .. math::
        \textrm{Can}(t_1, t_2, t_3) = e^{- i \frac{\pi}{2}(t_1 XX + t_2 YY + t_3 ZZ)}
    """

    def __init__(self, theta1, theta2, theta3, *args, **kwargs):
        super().__init__(linalg.expm(-1j / 2 * (theta1 * np.kron(X.data, X.data) +
                                                theta2 * np.kron(Y.data, Y.data) +
                                                theta3 * np.kron(Z.data, Z.data))),
                         name='Can', angles=(theta1, theta2, theta3), *args, **kwargs)


# Non-operation Gate

class Barrier:
    def __init__(self):
        raise NotImplementedError


class Measurement:
    def __init__(self):
        raise NotImplementedError


X = XGate()
Y = YGate()
Z = ZGate()
I = IGate()
S = SGate()
SDG = SDGGate()
T = TGate()
TDG = TDGGate()
H = HGate()
SWAP = SWAPGate()
ISWAP = ISWAPGate()

Can = WeylGate  # its alias: Canonical gate

PAULI_ROTATION_GATES = ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz']
ROTATION_GATES = ['rx', 'ry', 'rz', 'u1', 'u2' 'u3', 'rxx', 'ryy', 'rzz', 'can']
FIXED_GATES = ['x', 'y', 'z', 'i', 'id', 'h', 's', 't', 'sdg', 'tdg', 'cx', 'cz', 'swap', 'ch']
CONTROLLABLE_GATES = ['x', 'y', 'z', 'h', 'swap', 'rx', 'ry', 'rz', 'u3']
HERMITIAN_GATES = ['x', 'y', 'z', 'h', 'swap']
# READABLE_GATES = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'swap']

RYY_DEF = """gate ryy(param0) q0,q1 {
    rx(pi/2) q0; 
    rx(pi/2) q1; 
    cx q0, q1; 
    rz(param0) q1;
    cx q0, q1; 
    rx(-pi/2) q0;
    rx(-pi/2) q1;
}"""

CAN_DEF_BY_CNOT = """gate can (param0, param1, param2) q0,q1 {
    u3(1.5*pi, 0.0, 1.5*pi) q0;
    u3(0.5*pi, 1.5*pi, 0.5*pi) q1;
    cx q0, q1;
    u3(1.5*pi, param0 + pi, 0.5*pi) q0;
    u3(pi, 0.0, param1 + pi) q1;
    cx q0, q1;
    u3(0.5*pi, 0.0, 0.5*pi) q0;
    u3(0.0, 1.5*pi, param2 + 0.5*pi) q1;
    cx q0, q1;
}"""

CAN_DEF_BY_ISING = """gate can (param0, param1, param2) q0,q1 {
    rxx(param0) q0, q1;
    ryy(param1) q0, q1;
    rzz(param2) q0, q1;
}"""
