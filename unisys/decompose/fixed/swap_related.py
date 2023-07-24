"""
Decomposition rules for fixed gates composed of SWAP.
"""
from unisys.basic import Gate, Circuit
from unisys.basic import gate


def swap_decompose(SWAP: Gate) -> Circuit:
    """
    Decompose SWAP gate.

    ──@──       ──X────●────X──
      │    ==>    │    │    │
    ──@──       ──●────X────●──

    Args:
        SWAP: SWAP gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(SWAP, gate.SWAPGate) and len(SWAP.tqs) == 2 and len(SWAP.cqs) == 0):
        raise ValueError("SWAP must be a two target SWAPGate")
    tq1, tq2 = SWAP.tqs
    return Circuit([
        gate.X.on(tq1, tq2),
        gate.X.on(tq2, tq1),
        gate.X.on(tq1, tq2),
    ])


def cswap_decompose(CSWAP: Gate, totally: bool = False) -> Circuit:
    """
    Decompose CSWAP gate.

    ──●──       ───────●───────
      │                │
    ──@──  ==>  ──X────●────X──
      │           │    │    │
    ──@──       ──●────X────●──

    Args:
        CSWAP: Controlled-SWAP gate.
        totally: If True, decompose CSWAP into only single-qubiat and two-qubit gates.

    Returns:
        Circuit: Decomposed circuit.

    """
    if not (isinstance(CSWAP, gate.SWAPGate) and len(CSWAP.tqs) == 2 and len(CSWAP.cqs) == 1):
        raise ValueError("CSWAP must be a one control two target SWAPGate")
    cq = CSWAP.cq
    tq1, tq2 = CSWAP.tqs
    circ1 = Circuit([gate.X.on(tq1, tq2)])
    ccx = gate.X.on(tq2, [cq, tq1])
    circ3 = Circuit([gate.X.on(tq1, tq2)])
    if totally:
        from unisys.decompose.fixed.pauli_related import ccx_decompose
        return circ1 + ccx_decompose(ccx) + circ3
    return circ1 + Circuit([ccx]) + circ3
