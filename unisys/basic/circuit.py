"""
Quantum Circuit
"""
import io
from functools import reduce
from typing import List, Tuple
from copy import deepcopy
from datetime import datetime
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from . import gate
from .gate import Gate
from unisys.utils.operator import tensor_slots, controlled_unitary_matrix, is_equiv_unitary


class Circuit(list):
    def __init__(self, gates: List[Gate]):
        super().__init__(gates)

    # def __getitem__(self, idx):
    #     return self._gates[idx]
    #
    # def __len__(self):
    #     return len(self._gates)
    def append(self, *gates):
        for g in gates:
            super().append(g)

    def __repr__(self):
        return 'Circuit(num_gates: {}, num_qubits: {}, with_measure: {})'.format(self.num_gates, self.num_qubits,
                                                                                 self.with_measure)

    def display(self, style='text'):
        """
        Display the quantum circuit.
        First transform it into QASM string, then construct a qiskit.QuantumCirtuit instance, then draw the circuit.

        Args:
            style: optional: 'mpl', 'text', 'latex'
        """
        qc = QuantumCircuit.from_qasm_str(self.to_qasm())
        qc.draw(style)

    def to_qasm(self, fname: str = None):
        """Convert to QSAM in form of string"""
        # TODO: extend this function

        circuit = deepcopy(self)
        output = QASMStringIO()
        output.write_header()
        n = self.num_qubits

        # no measurement, just computing gates
        output.write_comment('Qubits: {}'.format(list(range(n))))
        output.write('qreg q[{}]'.format(n))
        output.write_line_gap()

        tuples_parsed = parse_to_tuples(circuit)
        output.write_comment('Quantum gate operations')
        for opr, idx in tuples_parsed:
            # if len(idx) == n:
            #     output.write_operation(opr, 'q')
            # else:
            output.write_operation(opr, 'q', *idx)

        qasm_str = output.getvalue()
        output.close()
        if fname is not None:
            with open(fname, 'w') as f:
                f.write(qasm_str)
        return qasm_str

    def unitary(self, msb: bool = False) -> np.ndarray:
        """
        Convert a quantum circuit to a unitary matrix.

        Args:
            msb (bool): if True, means the most significant bit (MSB) is on the left, i.e., little-endian representation

        Returns:
            Matrix, Equivalent unitary matrix representation.
        """
        ops = []
        n = self.num_qubits
        for g in self:
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

        def sort_gates_on_qreg(circuit: List[Gate], descend=False) -> List[Gate]:
            if descend:
                return sorted(circuit, key=lambda g: max(g.qregs))
            else:
                return sorted(circuit, key=lambda g: min(g.qregs))

        layers = []
        # building on parameter circuit and a corresponding DAG
        dag = self.to_dag()
        while len(dag.nodes) > 0:
            indices_font = _obtain_font_idx(dag)
            layers.append([self[idx] for idx in indices_font])
            dag.remove_nodes_from(indices_font)
        layers = list(map(sort_gates_on_qreg, layers))
        return layers

    def to_dag(self) -> nx.MultiDiGraph:
        """Convert a circuit into a Directed Acyclic Graph (DAG)"""
        circ_tmp = deepcopy(self)
        num_gates = len(circ_tmp)
        dag = nx.MultiDiGraph()
        dag.add_nodes_from(range(num_gates))
        while len(circ_tmp) > 0:
            gate_back = circ_tmp.pop()
            idx_back = len(circ_tmp)
            qreg_back = set(gate_back.qregs)
            union_set = set()
            for idx_font in range(idx_back - 1, -1, -1):
                gate_font = circ_tmp[idx_font]
                # judge if there is dependent relation about qubit(s) acted
                qreg_font = set(gate_font.qregs)
                if len(qreg_font & qreg_back) > 0 and len(qreg_font & union_set) == 0:
                    dag.add_edge(idx_font, idx_back)
                    union_set = union_set | qreg_font
        return dag

    @property
    def num_gates(self):
        return len(self)

    @property
    def with_measure(self):
        for g in self:
            if isinstance(g, gate.Measurement):
                return True
        return False

    @property
    def num_qubits(self):
        """number of qubits in the quantum circuit"""
        idx = []
        for g in self:
            idx.extend(g.tqs + g.cqs)
        return max(idx) + 1


class QASMStringIO(io.StringIO):
    """
    Specific StringIO extension class for QASM string generation
    """

    def __init__(self, *args, **kwargs):
        super(QASMStringIO, self).__init__(*args, **kwargs)

    def write(self, __s: str) -> int:
        n_content = super(QASMStringIO, self).write(__s)
        n_tail = super(QASMStringIO, self).write(';\n')
        return n_content + n_tail

    def write_operation(self, opr: str, qreg_name: str, *args) -> int:
        """
        Write computational gate operation into the string stream.
        :param opr: e.g. 'cx'
        :param qreg_name: e.g. 'q'
        :param args: e.g. 0, 1
        """
        if len(args) == 0:
            line = opr + ' ' + qreg_name + ';\n'
        else:
            line_list_qubits = []
            for idx in args:
                line_list_qubits.append(qreg_name + '[{}]'.format(idx))
            line = opr + ' ' + ', '.join(line_list_qubits) + ';\n'
        n = super(QASMStringIO, self).write(line)
        return n

    def write_line_gap(self, n: int = 1) -> int:
        n = super(QASMStringIO, self).write('\n' * n)
        return n

    def write_comment(self, comment: str) -> int:
        n = super(QASMStringIO, self).write('// ' + comment + '\n')
        return n

    def write_header(self) -> int:
        """
        Write the QASM text file header
        """
        n1 = super(QASMStringIO, self).write('// Author: zhy\n')
        n2 = super(QASMStringIO, self).write('// Time: {}\n'.format(datetime.now()))
        n3 = self.write_line_gap()
        n4 = super(QASMStringIO, self).write('OPENQASM 2.0;\n')
        n5 = super(QASMStringIO, self).write('include "qelib1.inc";\n')
        n6 = self.write_line_gap(2)
        return n1 + n2 + n3 + n4 + n5 + n6


def _limit_angle(a):
    """
    Limit equivalent rotation angle into (-pi, pi]
    """
    if a >= 0:
        r = a % (2 * np.pi)
        if r >= 0 and r <= np.pi:
            return r
        else:
            return r - 2 * np.pi
    else:
        r = (-a) % (2 * np.pi)
        if r >= 0 and r <= np.pi:
            return -r
        else:
            return 2 * np.pi - r


def _obtain_font_idx(dag: nx.MultiDiGraph) -> List[int]:
    """
    Obtain the font layer of a DAG
    """
    font_idx = []
    for gate in dag.nodes:
        if dag.in_degree(gate) == 0:
            font_idx.append(gate)
    return font_idx


def parse_to_tuples(circuit: Circuit) -> List[Tuple[str, List[int]]]:
    """
    Parse each Gate instance into a tuple consisting gate name and quantum register indices of a list
    :param circuit:
    :return: [('u3', [0]), ..., ('cu3', [0, 1])]
    """
    parsed_list = []
    for g in circuit:
        gname = g.name.lower()
        if gname in gate.fixed_gates:
            parsed_list.append((gname, g.qregs))
        elif gname in ['u3', 'cu3']:
            angles = list(map(_limit_angle, g.angles))
            opr = '{}({:.2f}, {:.2f}, {:.2f})'.format(gname, *angles)
            parsed_list.append((opr, g.qregs))
        else:
            # (C)R(x/y/z) gate
            angle = _limit_angle(g.angle)
            opr = '{}({:.2f})'.format(gname, angle)
            parsed_list.append((opr, g.qregs))
    return parsed_list


import cirq

cirq.Circuit


def optimize_circuit(circuit: Circuit) -> Circuit:
    """
    Optimize the quantum circuit, i.e., removing identity operators.

    Args:
        circuit (Circuit): original input circuit.

    Returns:
        Circuit, the optimized quantum circuit.
    """
    circuit_opt = Circuit()
    for g in circuit:
        if not is_equiv_unitary(g.data, gate.I.data):
            circuit_opt.append(g)
    return circuit_opt
# # TODO: rewrite QASM parser
