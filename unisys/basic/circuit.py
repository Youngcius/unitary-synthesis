"""
Quantum Circuit
"""
import io
from functools import reduce
from typing import List, Tuple, Dict
from copy import deepcopy
from datetime import datetime
import numpy as np
import networkx as nx
from unisys.basic import gate
from unisys.basic.gate import Gate
from unisys.utils.operator import tensor_slots, controlled_unitary_matrix, is_equiv_unitary
from unisys.utils.functions import limit_angle


class Circuit(list):
    def __init__(self, gates: List[Gate] = None):
        super().__init__(gates if gates else [])

    def append(self, *gates):
        for g in gates:
            super().append(g)

    def clone(self):
        return Circuit(deepcopy(self))

    def __add__(self, other):
        return Circuit(deepcopy(self.gates) + deepcopy(other.gates))

    def __repr__(self):
        return 'Circuit(num_gates: {}, num_qubits: {}, with_measure: {})'.format(self.num_gates, self.num_qubits,
                                                                                 self.with_measure)

    def to_qiskit(self):
        """Convert to qiskit.QuantumCircuit instance"""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            raise ImportError('qiskit is not installed')
        return QuantumCircuit.from_qasm_str(self.to_qasm())

    def to_cirq(self):
        """Convert to cirq.Circuit instance"""
        try:
            from cirq.contrib.qasm_import import circuit_from_qasm
        except ImportError:
            raise ImportError('cirq is not installed')
        return circuit_from_qasm(self.to_qasm())

    @classmethod
    def from_qasm(self, qasm_str: str = None, fname: str = None):
        """Convert QASM string to Circuit instance"""
        if qasm_str is None and fname is None:
            raise ValueError("Either qasm_str or fname should be given")
        if qasm_str is None:
            with open(fname, 'r') as f:
                qasm_str = f.read()
        raise NotImplementedError

    def to_qasm(self, fname: str = None):
        """Convert self to QSAM string"""
        circuit = deepcopy(self)
        output = QASMStringIO()
        output.write_header()
        n = self.num_qubits_with_dummy

        # no measurement, just computing gates
        output.write_comment('Qubits: {}'.format(list(range(n))))
        output.write('qreg q[{}]'.format(n))
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
        raise NotImplementedError

    def unitary(self, msb: bool = False) -> np.ndarray:
        """
        Convert a quantum circuit to a unitary matrix.

        Args:
            msb (bool): if True, means the most significant bit (MSB) is on the left, i.e., little-endian representation

        Returns:
            Matrix, Equivalent unitary matrix representation.
        """
        ops = []
        # n = self.num_qubits
        n = self.num_qubits_with_dummy
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
            indices_front = _obtain_front_indices(dag)
            layers.append([self[idx] for idx in indices_front])
            dag.remove_nodes_from(indices_front)
        layers = list(map(sort_gates_on_qreg, layers))
        return layers

    def to_dag(self) -> nx.MultiDiGraph:
        """Convert a circuit into a Directed Acyclic Graph (DAG) according to dependency of each gate's qubits"""
        all_gates = self.gates
        dag = nx.MultiDiGraph()
        dag.add_nodes_from(range(self.num_gates))
        while all_gates:
            g = all_gates.pop(0)
            qregs = set(g.qregs)
            if not all_gates:
                break
            for g_opt in all_gates:  # traverse the subsequent optional gates
                qregs_opt = set(g_opt.qregs)
                if qregs_opt & qregs:
                    dag.add_edge(self.index(g), self.index(g_opt))
                    qregs -= qregs_opt
                if not qregs:
                    break
        return dag

    def to_dag_forward(self) -> nx.MultiDiGraph:
        """
        Convert a circuit into a Directed Acyclic Graph (DAG) by forward traversing the circuit.
        ---
        NOTE: In comparison with the method `to_dag`, the DAG generated from this method is more compact
        """
        all_gates = self.gates
        dag = nx.MultiDiGraph()
        dag.add_nodes_from(range(self.num_gates))
        while all_gates:
            gate_front = all_gates.pop(0)
            qreg_front = set(gate_front.qregs)
            union_set = set()
            for gate_back in all_gates:
                # judge if there is dependent relation about qubit(s) acted
                qreg_back = set(gate_back.qregs)
                if len(qreg_back & qreg_front) > 0 and len(qreg_back & union_set) == 0:
                    dag.add_edge(self.index(gate_front), self.index(gate_back))
                    union_set = union_set | qreg_back
        return dag

    def to_dag_backward(self) -> nx.MultiDiGraph:
        """
        Convert a circuit into a Directed Acyclic Graph (DAG) by backward traversing the circuit.
        ---
        NOTE: In comparison with the method `to_dag`, the DAG generated from this method is more compact
        """
        all_gates = self.gates
        dag = nx.MultiDiGraph()
        dag.add_nodes_from(range(self.num_gates))
        while all_gates:
            gate_back = all_gates.pop()
            idx_back = len(all_gates)
            qreg_back = set(gate_back.qregs)
            union_set = set()
            for idx_front in range(idx_back - 1, -1, -1):
                gate_front = self[idx_front]
                # judge if there is dependent relation about qubit(s) acted
                qreg_front = set(gate_front.qregs)
                if len(qreg_front & qreg_back) > 0 and len(qreg_front & union_set) == 0:
                    dag.add_edge(idx_front, idx_back)
                    union_set = union_set | qreg_front
        return dag

    @property
    def gates(self, with_measure: bool = True):
        if with_measure:
            return [g for g in self]
        return [g for g in self if not isinstance(g, gate.Measurement)]

    @property
    def num_gates(self):
        return len(self)

    @property
    def qubits(self):
        """qubits indices in the quantum circuit"""
        idx = []
        for g in self:
            idx.extend(g.qregs)
        return list(set(idx))

    @property
    def num_qubits(self):
        """number of qubits in the quantum circuit (actually used)"""
        return len(self.qubits)

    @property
    def num_qubits_with_dummy(self):
        """number of qubits in the quantum circuit (including dummy qubits)"""
        return max(self.qubits) + 1

    @property
    def with_measure(self):
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

    def write(self, __s: str) -> int:
        n_content = super().write(__s)
        n_tail = super().write(';\n')
        return n_content + n_tail

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
        n1 = super().write('// Author: zhy\n')
        n2 = super().write('// Time: {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        n3 = self.write_line_gap()
        n4 = super().write('OPENQASM 2.0;\n')
        n5 = super().write('include "qelib1.inc";\n')
        n6 = self.write_line_gap(2)
        return n1 + n2 + n3 + n4 + n5 + n6


def _obtain_front_indices(dag: nx.MultiDiGraph) -> List[int]:
    """
    Obtain the front layer of a DAG
    """
    front_indices = []
    for node in dag.nodes:
        if dag.in_degree(node) == 0:
            front_indices.append(node)
    return front_indices


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
        if not ((g.n_qubits == 1 and len(g.cqs) <= 1) or (
                len(g.tqs) == 2 and len(g.cqs) <= 1 and isinstance(g, gate.SWAPGate))):
            raise ValueError('Only support 1Q or 2Q gates with designated qubits')
        gname = g.name.lower()

        if g.cqs:
            if gname not in gate.CONTROLLABLE_GATES:
                raise ValueError(f'{g} is not supported for transformation')
            if gname in gate.FIXED_GATES:
                opr = 'c{}'.format(gname)
            elif gname == 'u3':
                angles = list(map(limit_angle, g.angles))
                opr = 'cu3({:.2f}, {:.2f}, {:.2f})'.format(*angles)
            else:  # CR(X/Y/Z) gate
                angle = limit_angle(g.angle)
                opr = 'c{}({:.2f})'.format(gname, angle)
        else:
            if gname in gate.FIXED_GATES:
                opr = gname
            elif gname == 'u3':
                angles = list(map(limit_angle, g.angles))
                opr = '{}({:.2f}, {:.2f}, {:.2f})'.format(gname, *angles)
            else:  # R(X/Y/Z) gate
                angle = limit_angle(g.angle)
                opr = '{}({:.2f})'.format(gname, angle)
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
        if not is_equiv_unitary(g.data, gate.I.data):
            circuit_opt.append(g)
    return circuit_opt
