"""
Decomposition rules for fixed gates composed of H, S and T.
"""
from math import pi
from unisys.basic import gate
from unisys.basic import Gate, Circuit


def ch_decompose(CH: Gate) -> Circuit:
    """
    Decompose controlled-H gate.

    ──●──       ─────────────────●───────────────────
      │    ==>                   │
    ──H──       ──S────H────T────X────T†────H────S†──

    Args:
        CH: Controlled-H gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CH, gate.HGate) and len(CH.cqs) == 1 and len(CH.tqs) == 1):
        raise ValueError("CH must be a one control one target HGate")
    cq = CH.cq
    tq = CH.tq
    return Circuit([
        gate.S.on(tq),
        gate.H.on(tq),
        gate.T.on(tq),
        gate.X.on(tq, cq),
        gate.T.on(tq).hermitian(),
        gate.H.on(tq),
        gate.S.on(tq).hermitian(),
    ])


def cs_decompose(CS: Gate) -> Circuit:
    """
    Decompose controlled-S gate.

    ──●──  ==>  ──T────●──────────●──
      │    ==>         │          │
    ──S──  ==>  ──T────X────T†────X──

    Args:
        CS: Controlled-S gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CS, gate.SGate) and len(CS.cqs) == 1 and len(CS.tqs) == 1):
        raise ValueError("CS must be a one control one target SGate")
    cq = CS.cq
    tq = CS.tq
    return Circuit([
        gate.T.on(tq),
        gate.T.on(cq),
        gate.X.on(tq, cq),
        gate.T.on(tq).hermitian(),
        gate.X.on(tq, cq),
    ])


def ct_decompose(CT: Gate) -> Circuit:
    """
    Decompose controlled-T gate.

    ──●──       ──PS(-π/8)───●────────────────●──
      │    ==>               │                │
    ──T──       ──RZ(π/8)────X────RZ(-π/8)────X──

    Args:
        CT: Controlled-T gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CT, gate.TGate) and len(CT.cqs) == 1 and len(CT.tqs) == 1):
        raise ValueError("CT must be a one control one target TGate")
    cq = CT.cq
    tq = CT.tq
    return Circuit([
        gate.PhaseShift(pi / 8).on(cq),
        gate.RZ(pi / 8).on(tq),
        gate.X.on(tq, cq),
        gate.RZ(- pi / 8).on(tq),
        gate.X.on(tq, cq),
    ])
