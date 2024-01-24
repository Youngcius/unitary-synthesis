"""
Preprocessing circuit information (e.g., decompose SWAP and Toffoli)
"""
import os
import yaml
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from rich.console import Console
from collections import Counter

console = Console()

benchmark_dpath = '.'
with open(os.path.join(benchmark_dpath, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

fnames = []
for dpath in config['circ_dpaths']:
    dpath = os.path.join(benchmark_dpath, dpath)
    fnames.extend([os.path.join(dpath, fname) for fname in os.listdir(dpath)])

for fname in fnames:
    circ_name = fname.split('/')[-1].split('.')[0]
    console.rule(circ_name)
    qc = QuantumCircuit.from_qasm_file(fname)
    ops = [circ_instr.operation for circ_instr in qc.data]
    gate_names_count = dict(Counter([op.name for op in ops]))
    num_gates_count = dict(Counter([op.num_qubits for op in ops]))

    dag = circuit_to_dag(qc)
    used_qubits = [q.index for q in qc.qubits if q not in dag.idle_wires()]

    if len(used_qubits) < qc.num_qubits:
        console.print('!!!!! {} !!!!! There is unused qubits ({}/{})'.format(circ_name, len(used_qubits), qc.num_qubits), style='bold red')
    if 3 in num_gates_count:
        qc = qc.decompose('ccx')
        qc.qasm(filename=fname)
        console.print('Decompose Toffoli gates from {}'.format(circ_name))
    if 'swap' in gate_names_count:
        qc = qc.decompose('swap')
        qc.qasm(filename=fname)
        console.print('Decompose SWAP gates from {}'.format(circ_name))
