import sys
import warnings
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

warnings.filterwarnings('ignore')


fname = sys.argv[1]
if not fname.endswith('.qasm'):
    fname += '.qasm'

qc = QuantumCircuit.from_qasm_file(fname)
dag = circuit_to_dag(qc)
layers = list(dag.layers())
used_qubits = [q.index for q in qc.qubits if q not in dag.idle_wires()]


print(qc)
print('# qubit: {}, # gate: {}, # 2Q gate: {}, depth: {}'.format(len(used_qubits), len(qc), qc.num_nonlocal_gates(), qc.depth()))
