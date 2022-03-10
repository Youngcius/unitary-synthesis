"""
In our demo module, the circuit consists of just a series of Gate instances
"""

import numpy as np
from functools import reduce
from utils import tensor_1_slot, is_equiv_unitary
import matplotlib.pyplot as plt
from gate import Gate
from typing import List, Tuple
from copy import deepcopy
import datetime
import io
import gate


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
        n1 = super(QASMStringIO, self).write('// Author: Zhaohui Yang\n')
        n2 = super(QASMStringIO, self).write('// Time: {}\n'.format(datetime.datetime.now()))
        n3 = self.write_line_gap()
        n4 = super(QASMStringIO, self).write('OPENQASM 2.0;\n')
        n5 = super(QASMStringIO, self).write('include "qelib1.inc";\n')
        n6 = self.write_line_gap(2)
        return n1 + n2 + n3 + n4 + n5 + n6


def limit_angle(a):
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


def parse_to_tuples(circuit: List[Gate]) -> List[Tuple[str, List[int]]]:
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
            angles = list(map(limit_angle, g.angles))
            opr = '{}({:.2f}, {:.2f}, {:.2f})'.format(gname, *angles)
            parsed_list.append((opr, g.qregs))
        else:
            # (C)R(x/y/z) gate
            angle = limit_angle(g.angle)
            opr = '{}({:.2f})'.format(gname, angle)
            parsed_list.append((opr, g.qregs))
    return parsed_list


def num_qregs_of_circuit(circuit: List[Gate]):
    """
    number of qubits in the quantum circuit
    """
    idx = []
    for g in circuit:
        idx.extend(g.qregs)
    return max(idx) + 1


def to_qasm(circuit: List[Gate], fname=None) -> str:
    circuit = deepcopy(circuit)
    output = QASMStringIO()
    output.write_header()
    n = num_qregs_of_circuit(circuit)

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


def disp_circuit(circuit: List[Gate], style=None):
    """
    Display the quantum circuit.
    First transform it into QASM string, then construct a qiskit.QuantumCirtuit instance, then draw the circuit.
    :param circuit: list of Gate instances
    :param style: optional: 'mpl', 'text', 'latex'
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_str(to_qasm(circuit))
    if style == 'mpl':
        qc.draw('mpl')
        plt.show()
    else:
        print(qc.draw())  # default style: text


def circuit_to_unitary(circuit: List[Gate]):
    """
    Calculate the unitary matrix base on a series of Gate instances
    :param circuit: executable Gate instances, i.e. CNOT and single-qubit gate
    :return: type: np.ndarray
    """
    ops = []
    for g in circuit:
        if g.cq is None:
            ops.append(tensor_1_slot(g.data, 2, g.tq))
        else:
            ops.append(g.data)
    return reduce(np.dot, reversed(ops))


def optimize_circuit(circuit: List[Gate]) -> List[Gate]:
    """
    Delete redundant identity gates
    """
    circuit = deepcopy(circuit)
    I = np.identity(2)
    circ_new = []
    for g in circuit:
        # if g.cq is None:
        #     # for single-qubit gate
        if is_equiv_unitary(g.data, I):
            continue
        else:
            circ_new.append(g)
    return circ_new
