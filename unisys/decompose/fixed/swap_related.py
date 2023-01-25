"""
Decomposition rules for fixed gates composed of SWAP.
"""
from unisys.basic import Gate, Circuit
from unisys.basic import gate


def swap_decompose(swap: Gate) -> Circuit:
    if not (isinstance(swap, gate.SWAPGate) and len(swap.tqs) == 2 and len(swap.cqs) == 0):
        raise TypeError("SWAP must be a two target SWAPGate")
    tq1, tq2 = swap.tqs
    return Circuit([
        gate.X.on(tq2, tq1),
        gate.X.on(tq1, tq2),
        gate.X.on(tq2, tq1),
    ])


def cswap_decompose(cswap: Gate) -> Circuit:
    # TODO
    ...


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
    decompose(gate.SWAP.on([0, 1]), swap_decompose)
