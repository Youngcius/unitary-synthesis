from unisys.basic import gate
from unisys.basic import Gate, Circuit


def ccx_decompose(CCX: Gate) -> Circuit:
    """
    Decompose Toffoli gate.

        ──●──       ───────●────────────────────●────T──────────X────T†────X──
          │                │                    │               │          │
        ──●──  ==>  ───────┼──────────●─────────┼──────────●────●────T─────●──
          │                │          │         │          │
        ──X──       ──H────X────T†────X────T────X────T†────X────T────H────────

    Args:
        CCX: Toffoli gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CCX, gate.XGate) and len(CCX.cqs) == 2 and len(CCX.tqs) == 1):
        raise TypeError("CCX must be a two control one target XGate")
    cq1, cq2 = CCX.cqs
    tq = CCX.tq
    return Circuit([
        gate.H.on(tq),
        gate.X.on(tq, cq1),
        gate.T.on(tq).hermitian(),
        gate.X.on(tq, cq2),
        gate.T.on(tq),
        gate.X.on(tq, cq1),
        gate.T.on(tq).hermitian(),
        gate.X.on(tq, cq2),
        gate.T.on(tq),
        gate.T.on(cq1),
        gate.X.on(cq1, cq2),
        gate.H.on(tq),
        gate.T.on(cq2),
        gate.T.on(cq1).hermitian(),
        gate.X.on(cq1, cq2),
    ])


def cy_decompose(CY: Gate) -> Circuit:
    """
    Decompose CY gate.

        ──●──       ────────●───────
          │    ==>          │
        ──Y──       ──S†────X────S──

    Args:
        CY: Controlled-Y gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CY, gate.YGate) and len(CY.cqs) == 1 and len(CY.tqs) == 1):
        raise TypeError("CY must be a one control one target YGate")
    cq = CY.cq
    tq = CY.tq
    return Circuit([
        gate.S.on(tq).hermitian(),
        gate.X.on(tq, cq),
        gate.S.on(tq),
    ])


def cz_decompose(CZ: Gate) -> Circuit:
    """
    Decompose CY gate.

        ──●──       ───────●───────
          │    ==>         │
        ──Z──       ──H────X────H──

    Args:
        CZ: Controlled-Z gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CZ, gate.ZGate) and len(CZ.cqs) == 1 and len(CZ.tqs) == 1):
        raise TypeError("CZ must be a one control one target ZGate")
    cq = CZ.cq
    tq = CZ.tq
    return Circuit([
        gate.H.on(tq),
        gate.X.on(tq, cq),
        gate.H.on(tq),
    ])


def crx_decompose(CRX: Gate) -> Circuit:
    """
    Decompose CRX gate.

    ─────●─────       ───────●────────────────●───────────────────
         │       ==>         │                │
    ───RX(1)───       ──S────X────RY(-1/2)────X────RY(1/2)────S†──

    Args:
        CRX: Controlled-RX gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CRX, gate.RX) and len(CRX.cqs) == 1 and len(CRX.tqs) == 1):
        raise TypeError("CRX must be a one control one target RXGate")
    cq = CRX.cq
    tq = CRX.tq
    return Circuit([
        gate.S.on(tq),
        gate.X.on(tq, cq),
        gate.RY(- CRX.angle / 2).on(tq),
        gate.X.on(tq, cq),
        gate.RY(CRX.angle / 2).on(tq),
        gate.S.on(tq).hermitian(),
    ])


def cry_decompose(CRY: Gate) -> Circuit:
    """
    Decompose CRY gate.

    ─────●─────       ─────────────●────────────────●──
         │       ==>               │                │
    ───RY(1)───       ──RY(1/2)────X────RY(-1/2)────X──

    Args:
        CRY: Controlled-RY gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CRY, gate.RY) and len(CRY.cqs) == 1 and len(CRY.tqs) == 1):
        raise TypeError("CRY must be a one control one target RYGate")
    cq = CRY.cq
    tq = CRY.tq
    return Circuit([
        gate.RY(CRY.angle / 2).on(tq),
        gate.X.on(tq, cq),
        gate.RY(-CRY.angle / 2).on(tq),
        gate.X.on(tq, cq),
    ])


def crz_decompose(CRZ: Gate) -> Circuit:
    """
    Decompose CRZ gate.

    ─────●─────       ─────────────●────────────────●──
         │       ==>               │                │
    ───RZ(1)───       ──RZ(1/2)────X────RZ(-1/2)────X──

    Args:
        CRZ: Controlled-RZ gate.

    Returns:
        Circuit: Decomposed circuit.
    """
    if not (isinstance(CRZ, gate.RZ) and len(CRZ.cqs) == 1 and len(CRZ.tqs) == 1):
        raise TypeError("CRZ must be a one control one target RZGate")
    cq = CRZ.cq
    tq = CRZ.tq
    return Circuit([
        gate.RZ(CRZ.angle / 2).on(tq),
        gate.X.on(tq, cq),
        gate.RZ(-CRZ.angle / 2).on(tq),
        gate.X.on(tq, cq),
    ])


def decompose(g, decomp_func):
    import cirq
    from unisys.utils.operator import tensor_slots, controlled_unitary_matrix
    from functools import reduce
    import numpy as np

    n = max(g.tqs + g.cqs) + 1

    if g.n_qubits > int(np.log2(g.data.shape[0])) == 1:
        data = reduce(np.kron, [g.data] * g.n_qubits)
    else:
        data = g.data

    if g.cqs:
        U = controlled_unitary_matrix(data, len(g.cqs))
        U = tensor_slots(U, n, g.cqs + g.tqs)
    else:
        U = tensor_slots(data, n, g.tqs)

    circ = decomp_func(g)
    print(circ)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(),
        U,
        atol=1e-5
    )


if __name__ == '__main__':
    decompose(gate.X.on([1], [0, 2]), ccx_decompose)
    decompose(gate.Y.on(1, 0), cy_decompose)
    decompose(gate.Z.on(1, 0), cz_decompose)
    decompose(gate.RX(1.2).on(2, 0), crx_decompose)
    decompose(gate.RY(1.2).on(2, 0), cry_decompose)
    decompose(gate.RZ(1.2).on(2, 0), crz_decompose)
