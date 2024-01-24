"""
Usage: python get_circ_info.py <circuit_file_name>
"""
import sys
import warnings
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from rich.console import Console
from rich.table import Table
from collections import Counter
console = Console()

warnings.filterwarnings('ignore')


fname = sys.argv[1]
if not fname.endswith('.qasm'):
    fname += '.qasm'

qc = QuantumCircuit.from_qasm_file(fname)
dag = circuit_to_dag(qc)
layers = list(dag.layers())
used_qubits = [q.index for q in qc.qubits if q not in dag.idle_wires()]

ops = [circ_instr.operation for circ_instr in qc.data]
gate_names_count = dict(Counter([op.name for op in ops]))
gate_names_count = dict(sorted(gate_names_count.items()))
num_gates_count = dict(Counter([op.num_qubits for op in ops]))
num_gates_count = dict(sorted(num_gates_count.items()))

# print(qc)
console.print(gate_names_count)
console.print(num_gates_count)
# console.print('num_qubits: {}, num_gates: {}, num_2q_gates: {}, depth: {}'.format(qc.num_qubits, len(qc), qc.num_nonlocal_gates(), qc.depth()))
# use rich Table
circ_name = fname.split('/')[-1].split('.')[0]
table = Table(title=circ_name)
table.add_column("num_qubits")
table.add_column("num_gates")
table.add_column("num_2q_gates")
table.add_column("depth")
table.add_row(str(qc.num_qubits), str(len(qc)), str(qc.num_nonlocal_gates()), str(qc.depth()))
console.print(table)
