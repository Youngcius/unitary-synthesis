"""
Quantum Circuit
"""
import io
import re
import os
import uuid
import numpy as np
import networkx as nx
from math import pi
from functools import reduce
from operator import add
from typing import List, Tuple, Dict
from copy import deepcopy, copy
from collections import Counter

from unisys.basic import gate
from unisys.basic.gate import Gate
from unisys.utils.operations import tensor_slots, controlled_unitary_matrix, is_equiv_unitary
from unisys.utils.functions import limit_angle
from unisys.utils.graphs import draw_circ_dag_mpl, draw_circ_dag_graphviz


class Circuit(list):
    def __init__(self, gates: List[Gate] = None):
        if gates is not None:
            assert all([g.qregs for g in gates]), "Each gate should act on specific qubit(s)"
        super().__init__(gates if gates else [])

    def __hash__(self):
        return hash(id(self))

    def append(self, *gates):
        assert all([g.qregs for g in gates]), "Each gate should act on specific qubit(s)"
        for g in gates:
            super().append(g)

    def prepend(self, *gates):
        assert all([g.qregs for g in gates]), "Each gate should act on specific qubit(s)"
        for g in reversed(gates):
            super().insert(0, g)

    def deepclone(self):
        """Deep duplicate"""
        return Circuit(deepcopy(self))

    def clone(self):
        """Shadow duplicate"""
        return Circuit(copy(self))

    def __add__(self, other):
        return Circuit(deepcopy(self.gates) + deepcopy(other.gates))

    def __repr__(self):
        return 'Circuit(num_gates: {}, num_qubits: {})'.format(self.num_gates, self.num_qubits)

    def to_qiskit(self):
        """Convert to qiskit.QuantumCircuit instance"""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            raise ImportError('qiskit is not installed')
        return QuantumCircuit.from_qasm_str(self.to_qasm())

    @classmethod
    def from_qiskit(cls, qiskit_circ):
        """Convert from qiskit.QuantumCircuit instance"""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            raise ImportError('qiskit is not installed')
        assert isinstance(qiskit_circ, QuantumCircuit), "Input should be a qiskit.QuantumCircuit instance"
        return cls.from_qasm(qiskit_circ.qasm())

    def to_cirq(self):
        """Convert to cirq.Circuit instance"""
        try:
            from cirq.contrib.qasm_import import circuit_from_qasm
        except ImportError:
            raise ImportError('cirq is not installed')
        return circuit_from_qasm(self.to_qasm())

    @classmethod
    def from_cirq(cls, cirq_circ):
        """Convert from cirq.Circuit instance"""
        try:
            import cirq
        except ImportError:
            raise ImportError('cirq is not installed')
        assert isinstance(cirq_circ, cirq.Circuit), "Input should be a cirq.Circuit instance"
        return cls.from_qasm(cirq.qasm(cirq_circ))

    def to_tket(self):
        """Convert to pytket.circuit.Circuit instance"""
        try:
            from pytket.circuit import Circuit
            from pytket.qasm import circuit_from_qasm_str
        except ImportError:
            raise ImportError('pytket is not installed')
        return circuit_from_qasm_str(self.to_qasm())

    @classmethod
    def from_tket(cls, tket_circ):
        """Convert from pytket.circuit.Circuit instance"""
        try:
            from pytket.circuit import Circuit
            from pytket.qasm import circuit_to_qasm_str
        except ImportError:
            raise ImportError('pytket is not installed')
        assert isinstance(tket_circ, Circuit), "Input should be a pytket.circuit.Circuit instance"
        return cls.from_qasm(circuit_to_qasm_str(tket_circ))

    def to_bqskit(self):
        """Convert to bqskit.Circuit instance"""
        try:
            import bqskit
        except ImportError:
            raise ImportError('bqskit is not installed')
        tmp_file = '{}.qasm'.format(uuid.uuid4())
        self.to_qasm(fname=tmp_file)
        bqskit_circ = bqskit.Circuit.from_file(tmp_file)
        os.remove(tmp_file)
        return bqskit_circ

    @classmethod
    def from_bqskit(cls, bqskit_circ):
        """Convert from bqskit.Circuit instance"""
        try:
            import bqskit
        except ImportError:
            raise ImportError('bqskit is not installed')
        assert isinstance(bqskit_circ, bqskit.Circuit), "Input should be a bqskit.Circuit instance"
        tmp_file = '{}.qasm'.format(uuid.uuid4())
        bqskit_circ.save(tmp_file)
        with open(tmp_file, 'r') as f:
            qasm = f.read()
        os.remove(tmp_file)
        return cls.from_qasm(qasm)

    def to_qasm(self, fname: str = None):
        """Convert self to QSAM string"""
        circuit = deepcopy(self)
        output = QASMStringIO()
        output.write_header()
        n = self.num_qubits_with_dummy

        if 'RYY' in self.gate_stats() or 'Can' in self.gate_stats():
            output.write_comment("Customized 'ryy' gate definitions")
            output.write(gate.RYY_DEF)
            output.write_line_gap(2)

        if 'Can' in self.gate_stats():
            output.write_comment("Customized 'can' gate definitions")
            output.write(gate.CAN_DEF_BY_ISING)  # alternatively, can use "gate.CAN_DEF_BY_CNOT"
            output.write_line_gap(2)

        # no measurement, just computing gates
        output.write_comment('Qubits: {}'.format(list(range(n))))
        output.write_qregs(n)
        # output.write('qreg q[{}]'.format(n))
        output.write_line_gap()

        tuples_parsed = parse_to_tuples(circuit)
        output.write_comment('Quantum gate operations')
        for opr, idx in tuples_parsed:
            output.write_operation(opr, 'q', *idx)

        qasm_str = output.getvalue()
        output.close()
        if fname is not None:
            with open(fname, 'w') as f:
                f.write(qasm_str)
        return qasm_str

    @classmethod
    def from_qasm(cls, qasm_str: str = None, fname: str = None):
        """
        Convert QASM string to Circuit instance

        First parse each line as a list of strings, e.g.,

        'cx q[2], q[1];',                     -->  ['cx', 'q[2]', 'q[1]']
        'h q[2];',                            -->  ['h', 'q[2]']
        'rx(0.50) q[0];',                     -->  ['rx', '0.50', 'q[0]']
        'ry(0.50) q[1];',                     -->  ['ry', '0.50', 'q[1]']
        'u3(0.30, 0.12, 1.12) q[2];',         -->  ['u3', '0.30', '0.12', '1.12', 'q[2]']
        'cu3(0.30, 0.12, 1.12) q[0], q[1];',  -->  ['cu3', '0.30', '0.12', '1.12', 'q[0]', 'q[1]']

        Then construct the corresponding Gate instance according to the parsed list.
        """
        if qasm_str is None and fname is None:
            raise ValueError("Either qasm_str or fname should be given")
        if qasm_str is None:
            with open(fname, 'r') as f:
                qasm_str = f.read()
        # print(qasm_str)
        input = io.StringIO(qasm_str)
        circ = Circuit()
        for line in input.readlines():
            if (line.startswith('//') or line.startswith('OPENQASM') or line.startswith('include') or
                    line.startswith('qreg') or line.startswith('creg') or line.startswith('barrier') or
                    line.startswith('gate') or line.startswith('{') or line.startswith('}') or line.startswith('  ') or
                    line.startswith('measure')):
                continue
            line = line.strip().strip(';')
            if line == '':
                continue
            line = re.split(r'\s*,\s*|\s+', line)  # split according to ',' or '\s+'
            line = [re.split(r'\(|\)', s) for s in line]  # split each element according to '(' or ')'
            line = reduce(add, line)
            line = [s for s in line if s != '']

            def parse_qubit_index(s):
                """e.g., qubits[10] --> 10; q[5] --> 5"""
                return int(re.findall(r'\d+', s)[0])

            # parse gname, tq, cq
            if line[0].startswith('c') and line[0] != 'can':
                gname = line[0][1:]
                tq = parse_qubit_index(line[-1])
                cq = parse_qubit_index(line[-2])
            else:
                gname = line[0]
                if gname in ['swap', 'rxx', 'ryy', 'rzz', 'can']:
                    tq = [parse_qubit_index(line[-2]), parse_qubit_index(line[-1])]
                else:
                    tq = parse_qubit_index(line[-1])
                cq = None

            if gname == 'id':
                gname = 'i'

            # create Gate instance
            if gname in gate.FIXED_GATES:
                g = getattr(gate, gname.upper()).on(tq, cq)
            elif gname in ['u3', 'can']:
                angles = [eval(line[1]), eval(line[2]), eval(line[3])]
                g = getattr(gate, gname.capitalize())(*angles).on(tq, cq)
            elif gname == 'u2':
                angles = [eval(line[1]), eval(line[2])]
                g = gate.U2(*angles).on(tq, cq)
            elif gname in ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'u1']:
                angle = eval(line[1])
                g = getattr(gate, gname.upper())(angle).on(tq, cq)
            else:
                raise ValueError(f'Unsupported gate {gname}')
            circ.append(g)

        return circ

    def rewire(self, mapping: Dict[int, int]):
        """
        Rewire the circuit according to a given mapping
        """
        mapped_circ = Circuit()
        for g in self:
            mapped_circ.append(g.on(
                [mapping[tq] for tq in g.tqs],
                [mapping[cq] for cq in g.cqs]
            ))
        return mapped_circ

    def inverse(self):
        """Inverse of the original circuit by reversing the order of gates' hermitian conjugates"""
        return Circuit([g.hermitian() for g in reversed(self)])

    def unitary(self, msb: bool = False, with_dummy: bool = False) -> np.ndarray:
        """
        Convert a quantum circuit to a unitary matrix.

        Args:
            msb (bool): if True, means the most significant bit (MSB) is on the left, i.e., little-endian representation
            with_dummy (bool): if True, means taking dummy qubits into account when calculating the unitary matrix

        Returns:
            Matrix, Equivalent unitary matrix representation.
        """
        ops = []
        if with_dummy:
            n = self.num_qubits_with_dummy
            circ = self
        else:
            n = self.num_qubits
            circ = self.rewire({p: q for p, q in zip(self.qubits, range(n))})
        for g in circ:
            if g.n_qubits > int(np.log2(g.data.shape[0])) == 1:
                # identical tensor-product gate expanded from single-qubit gate
                data = reduce(np.kron, [g.data] * g.n_qubits)
            else:
                data = g.data

            if not g.cqs:
                mat = tensor_slots(data, n, g.tqs)
            else:
                mat = controlled_unitary_matrix(data, len(g.cqs))
                mat = tensor_slots(mat, n, g.cqs + g.tqs)

            ops.append(mat)

        unitary = reduce(np.dot, reversed(ops))
        if msb:
            unitary = tensor_slots(unitary, n, list(range(n - 1, -1, -1)))
        return unitary

    def layer(self) -> List[List[Gate]]:
        """Divide a circuit into different layers"""
        from unisys.utils.passes import dag_to_layers

        layers = dag_to_layers(self.to_dag())
        layers = list(map(_sort_gates_on_qreg, layers))
        return layers


    def to_dag(self) -> nx.DiGraph:
        """Convert a circuit into a Directed Acyclic Graph (DAG) according to dependency of each gate's qubits"""
        all_gates = self.gates
        dag = nx.DiGraph()
        dag.add_nodes_from(all_gates)
        while all_gates:
            g = all_gates.pop(0)
            qregs = set(g.qregs)
            for g_opt in all_gates:  # traverse the subsequent optional gates
                qregs_opt = set(g_opt.qregs)
                if qregs_opt & qregs:
                    dag.add_edge(g, g_opt)
                    qregs -= qregs_opt
                if not qregs:
                    break
        return dag

    def draw_circ_dag_mpl(self, fname: str = None, figsize=None, fix_layout=False):
        return draw_circ_dag_mpl(self.to_dag(), fname, figsize, fix_layout)

    def draw_circ_dag_graphviz(self, fname: str = None):
        return draw_circ_dag_graphviz(self.to_dag(), fname)

    def gate_stats(self) -> Dict[str, int]:
        """Statistics of gate names and occurring number in the circuit"""
        return dict(sorted(Counter([g.name for g in self.gates]).items()))

    @property
    def gates(self, with_measure: bool = True) -> List[Gate]:
        if with_measure:
            return [g for g in self]
        return [g for g in self if not isinstance(g, gate.Measurement)]

    @property
    def num_gates(self):
        return len(self)

    @property
    def num_nonlocal_gates(self):
        return len([g for g in self if g.num_qregs > 1])

    @property
    def depth(self):
        """number of layers"""
        return len(self.layer())

    @property
    def depth_nonlocal(self):
        """number of layers including nonlocal gates"""
        layers = self.layer()
        return len([l for l in layers if Circuit(l).num_nonlocal_gates > 0])

    @property
    def qubits(self) -> List[int]:
        """qubits indices in the quantum circuit"""
        idx = []
        for g in self:
            idx.extend(g.qregs)
        return sorted(list(set(idx)))

    @property
    def num_qubits(self):
        """number of qubits in the quantum circuit (actually used)"""
        return len(self.qubits)

    @property
    def num_qubits_with_dummy(self):
        """number of qubits in the quantum circuit (including dummy qubits)"""
        return max(self.qubits) + 1

    @property
    def max_gate_weight(self):
        """maximum gate weight"""
        return max([g.num_qregs for g in self])

    @property
    def with_measure(self):
        """Whether the circuit contains measurement gates"""
        for g in self:
            if isinstance(g, gate.Measurement):
                return True
        return False


class QASMStringIO(io.StringIO):
    """
    Specific StringIO extension class for QASM string generation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def write(self, __s: str) -> int:
    #     n_content = super().write(__s)
    #     n_tail = super().write(';\n')
    #     return n_content + n_tail
    
    def write_qregs(self, n: int) -> int:
        """
        Write quantum register declarations into the string stream.

        Args:
            n: number of qubits
        """
        n = super().write('qreg q[{}];\n'.format(n))
        return n

    def write_operation(self, opr: str, qreg_name: str, *args) -> int:
        """
        Write computational gate operation into the string stream.

        Args:
            opr: e.g. 'cx'
            qreg_name: e.g. 'q'
            args: e.g. 0, 1
        """
        if len(args) == 0:
            line = opr + ' ' + qreg_name + ';\n'
        else:
            line_list_qubits = []
            for idx in args:
                line_list_qubits.append(qreg_name + '[{}]'.format(idx))
            line = opr + ' ' + ', '.join(line_list_qubits) + ';\n'
        n = super().write(line)
        return n

    def write_line_gap(self, n: int = 1) -> int:
        n = super().write('\n' * n)
        return n

    def write_comment(self, comment: str) -> int:
        n = super().write('// ' + comment + '\n')
        return n

    def write_header(self) -> int:
        """
        Write the QASM text file header
        """
        # n1 = super().write('// Author: zhy\n')
        # n2 = super().write('// Time: {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # n3 = self.write_line_gap()
        n4 = super().write('OPENQASM 2.0;\n')
        n5 = super().write('include "qelib1.inc";\n')
        n6 = self.write_line_gap(2)
        # return n1 + n2 + n3 + n4 + n5 + n6
        return n4 + n5 + n6


def parse_to_tuples(circuit: Circuit) -> List[Tuple[str, List[int]]]:
    """
    Parse each Gate instance into a tuple consisting gate name and quantum register indices of a list

    Args:
        circuit (Circuit): input Circuit instance

    Returns:
        List of tuples representing designated quantum operation, e.g. [('u3', [0]), ..., ('cu3', [0, 1])]
    """
    parsed_list = []
    for g in circuit:
        if not ((g.n_qubits == 1 and len(g.cqs) <= 1) or
                (len(g.tqs) == 2 and len(g.cqs) <= 1 and isinstance(g, gate.SWAPGate)) or
                (len(g.tqs) == 2 and len(g.cqs) == 0 and isinstance(g, (gate.RXX, gate.RYY, gate.RZZ, gate.Can)))):
            raise ValueError('Only support 1Q or 2Q gates with designated qubits')

        if g.name == 'I':
            gname = 'id'
        else:
            gname = g.name.lower()

        if g.cqs:
            if gname not in gate.CONTROLLABLE_GATES:
                raise ValueError(f'{g} is not supported for transformation')
            if gname in gate.FIXED_GATES:
                opr = 'c{}'.format(gname)
            elif gname == 'u3':
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                # opr = 'cu3({:.4f}, {:.4f}, {:.4f})'.format(*angles)
                opr = 'cu3({}*pi, {}*pi, {}*pi)'.format(*factors)
            elif gname == 'u2':
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                opr = 'cu2({}*pi, {}*pi)'.format(*factors)
            else:  # CR(X/Y/Z) and U1 gate
                angle = limit_angle(g.angle)
                factor = angle / pi
                opr = 'c{}({}*pi)'.format(gname, factor)
        else:
            if gname in gate.FIXED_GATES:
                opr = gname
            elif gname in ['u3', 'can']:
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                opr = '{}({}*pi, {}*pi, {}*pi)'.format(gname, *factors)
            elif gname == 'u2':
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                opr = '{}({}*pi, {}*pi)'.format(gname, *factors)
            else:  # R(X/Y/Z), R(XX/YY/ZZ), U1
                angle = limit_angle(g.angle)
                factor = angle / pi
                opr = '{}({}*pi)'.format(gname, factor)
        parsed_list.append((opr, g.qregs))
    return parsed_list


def optimize_circuit(circuit: Circuit) -> Circuit:
    """
    Optimize the quantum circuit, i.e., removing identity operators.
    Naive strategy: remove all identity operators.

    Args:
        circuit (Circuit): original input circuit.

    Returns:
        Circuit, the optimized quantum circuit.
    """
    circuit_opt = Circuit()
    for g in circuit:
        if not (g.num_qregs == 1 and is_equiv_unitary(g.data, gate.I.data)):
            circuit_opt.append(g)
    return circuit_opt


def _sort_gates_on_qreg(circuit: List[Gate], descend=False) -> List[Gate]:
    if descend:
        return sorted(circuit, key=lambda g: max(g.qregs))
    else:
        return sorted(circuit, key=lambda g: min(g.qregs))
