"""
Decomposition rules for fixed gates composed of H, S and T.
"""
from math import pi
from unisys.basic import gate
from unisys.basic import Gate, Circuit
from unisys.utils.operator import controlled_unitary_matrix


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
        raise TypeError("CH must be a one control one target HGate")
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
        raise TypeError("CS must be a one control one target SGate")
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
        raise TypeError("CT must be a one control one target TGate")
    cq = CT.cq
    tq = CT.tq
    return Circuit([
        gate.PhaseShift(pi / 8).on(cq),
        gate.RZ(pi / 8).on(tq),
        gate.X.on(tq, cq),
        gate.RZ(- pi / 8).on(tq),
        gate.X.on(tq, cq),
    ])


if __name__ == '__main__':
    import cirq

    ch = gate.H.on(1, 0)
    circ = ch_decompose(ch)
    print(circ)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        controlled_unitary_matrix(ch.data),
        atol=1e-5
    )

    cs = gate.S.on(1, 0)
    circ = cs_decompose(cs)
    print(circ)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        controlled_unitary_matrix(cs.data),
        atol=1e-5
    )

    ct = gate.T.on(1, 0)
    circ = ct_decompose(ct)
    print(circ)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        controlled_unitary_matrix(ct.data),
        atol=1e-5
    )
