"""
Approximate synthesis using CNOT + U3 gates, or Canonical + U3 gates
"""

from unisys.basic.circuit import Circuit
import bqskit
from bqskit.ir.gates import CanonicalGate, CNOTGate, U3Gate


def approx_to_cnot(circ: Circuit, max_synthesis_size: int = 3, optimization_level: int = 1) -> Circuit:
    """
    Approximate synthesis using CNOT + U3 gates

    Args:
        circ: The input circuit
        max_synthesis_size: The maximum size of a unitary to synthesize or instantiate. Larger circuits will be partitioned.
            Increasing this will most likely lead to better results with an exponential time trade-off. (Default: 3)
        optimization_level: The optimization level to use. (Default: 1)
    """
    circ_bqs = circ.to_bqskit()
    model = bqskit.MachineModel(circ.num_qubits_with_dummy, gate_set={CNOTGate(), U3Gate()})
    circ_bqs_opt = bqskit.compile(
        circ_bqs, model, max_synthesis_size=max_synthesis_size, optimization_level=optimization_level)
    circ_opt = Circuit.from_bqskit(circ_bqs_opt)
    return circ_opt


def approx_to_su4(circ: Circuit, max_synthesis_size: int = 3, optimization_level: int = 1) -> Circuit:
    """
    Approximate synthesis using SU(4) gates, i.e., Canonical + U3

    Args:
        circ: The input circuit
        max_synthesis_size: The maximum size of a unitary to synthesize or instantiate. Larger circuits will be partitioned.
            Increasing this will most likely lead to better results with an exponential time trade-off. (Default: 3)
        optimization_level: The optimization level to use. (Default: 1)
    """
    circ_bqs = circ.to_bqskit()
    model = bqskit.MachineModel(circ.num_qubits_with_dummy, gate_set={CanonicalGate(), U3Gate()})
    circ_bqs_opt = bqskit.compile(
        circ_bqs, model, max_synthesis_size=max_synthesis_size, optimization_level=optimization_level)
    circ_opt = Circuit.from_bqskit(circ_bqs_opt)
    return circ_opt
