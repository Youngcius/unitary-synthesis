"""
Template-based compiling passes
"""
from unisys.basic.circuit import Circuit


def paulisimp(circ: Circuit, epoch: int = 1) -> Circuit:
    """
    Apply PauliSimp() from pytket to simplify Pauli gadgets.
    """
    from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
    from pytket.passes import PauliSimp, PeepholeOptimise2Q

    circ_tket = circuit_from_qasm_str(circ.to_qasm())
    for _ in range(epoch):
        PauliSimp().apply(circ_tket)
    PeepholeOptimise2Q().apply(circ_tket)
    return Circuit.from_qasm(circuit_to_qasm_str(circ_tket))


def adaptive_paulisimp(circ: Circuit) -> Circuit:
    """
    Apply PauliSimp() iteratively until no more simplification can be done (metric is circuit depth).
    """
    from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
    from pytket.passes import PauliSimp, PeepholeOptimise2Q

    circ_tket = circuit_from_qasm_str(circ.to_qasm())
    circ_tket_copy = circ_tket.copy()
    best_depth = circ_tket.depth()
    while True:
        PauliSimp().apply(circ_tket_copy)
        if best_depth > circ_tket_copy.depth():
            best_depth = circ_tket_copy.depth()
            circ_tket = circ_tket_copy.copy()
        else:
            break
    PeepholeOptimise2Q().apply(circ_tket)
    return Circuit.from_qasm(circuit_to_qasm_str(circ_tket))
