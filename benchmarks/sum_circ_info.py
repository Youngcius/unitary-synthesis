"""
Summarize circuit information (e.g., # qubits, # gates, depth, etc.) of all benchmark programs
"""
import os
import yaml
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from rich.console import Console
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

console = Console()

benchmark_dpath = '.'
with open(os.path.join(benchmark_dpath, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

fnames = []
for dpath in config['circ_dpaths']:
    dpath = os.path.join(benchmark_dpath, dpath)
    fnames.extend([os.path.join(dpath, fname) for fname in os.listdir(dpath)])


description = pd.DataFrame(columns=['circ_name', 'num_qubits', 'num_gates', 'num_2q_gates', 'depth'])

all_gate_names = []
for i, fname in enumerate(fnames):
    circ_name = fname.split('/')[-1].split('.')[0]
    qc = QuantumCircuit.from_qasm_file(fname)
    dag = circuit_to_dag(qc)
    used_qubits = [q.index for q in qc.qubits if q not in dag.idle_wires()]
    ops = [circ_instr.operation for circ_instr in qc.data]
    gate_names_count = dict(Counter([op.name for op in ops]))
    gate_names_count = dict(sorted(gate_names_count.items()))
    num_gates_count = dict(Counter([op.num_qubits for op in ops]))
    num_gates_count = dict(sorted(num_gates_count.items()))

    console.rule(circ_name)
    console.print(gate_names_count)
    console.print('gate weight stats: {}'.format(num_gates_count))

    all_gate_names += list(gate_names_count.keys())

    if len(used_qubits) < qc.num_qubits:
        console.print('!!!!! {} !!!!! There is unused qubits'.format(circ_name), style='bold red')
    if 'swap' in gate_names_count:
        console.print('!!!!! {} !!!!! There is SWAP gate'.format(circ_name), style='bold red')
    if 3 in num_gates_count:
        console.print('!!!!! {} !!!!! There is 3-qubit gate'.format(circ_name), style='bold red')

    description = pd.concat([description, pd.DataFrame({
        'circ_name': circ_name,
        'num_qubits': qc.num_qubits,
        'num_gates': qc.size(),
        'num_2q_gates': qc.num_nonlocal_gates(),
        'depth': qc.depth()
    }, index=[0])], ignore_index=True)


print()
console.print('All gates occurring in benchmarks: {}'.format(np.unique(all_gate_names)))
print()

console.print(description)

description.to_csv('description.csv', index=False)
