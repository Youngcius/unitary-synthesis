import numpy as np
from typing import List
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate as QuantumGate
from qiskit.circuit.library.standard_gates import (XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate,
                                                   RXGate, RYGate, RZGate, CXGate, CYGate, CZGate, CHGate,
                                                   CCXGate, CSwapGate)

from qiskit.circuit.random import random_circuit

def gene_random_circuit(num_qubits: int, depth: int, seed: int = None,
                        one_q_ops: List[QuantumGate] = [XGate, HGate, SGate, SdgGate, TGate, TdgGate],
                        two_q_ops: List[QuantumGate] = [CXGate],
                        three_q_ops: List[QuantumGate] = [CCXGate]) -> QuantumCircuit:
    """
    Generate random circuit of arbitrary size and form.

    This function leverages traits from qiskit.circuit.random.random_circuit function.


    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        seed (int): sets random seed (optional)

    Returns:
        Circuit: constructed circuit
    """
    max_operands = 2
    # one_param = [RXGate, RYGate, RZGate]
    one_param = []
    two_param = []
    three_param = []

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)

        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            qc.append(op, register_operands)

    return qc


if __name__ == '__main__':
    for n in [8, 16, 32, 64]:
        qc = gene_random_circuit(n, n * 10, seed=123)

        print('generated random circuit of {} qubits'.format(n))
        qc.qasm(filename='applications/rand_{}.qasm'.format(n))
