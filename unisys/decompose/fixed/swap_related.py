"""
Decomposition rules for fixed gates composed of SWAP.
"""
from unisys.basic import Gate, Circuit
from unisys.basic import gate


def swap_decompose(swap: Gate) -> Circuit:
    if not (isinstance(swap, gate.SWAPGate) and len(swap.tqs) == 2 and len(swap.cqs) == 0):
        raise ValueError("SWAP must be a two target SWAPGate")
    tq1, tq2 = swap.tqs
    return Circuit([
        gate.X.on(tq2, tq1),
        gate.X.on(tq1, tq2),
        gate.X.on(tq2, tq1),
    ])


def cswap_decompose(cswap: Gate) -> Circuit:
    if not (isinstance(cswap, gate.SWAPGate) and len(cswap.tqs) == 2 and len(cswap.cqs) == 1):
        raise ValueError("CSWAP must be a one control two target SWAPGate")
    raise NotImplementedError
