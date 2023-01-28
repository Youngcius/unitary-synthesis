def assert_equivalent_unitary(U, V):
    try:
        import cirq
        cirq.testing.assert_allclose_up_to_global_phase(U, V, atol=1e-5)
    except ModuleNotFoundError:
        from unisys.utils.operator import is_equiv_unitary
        assert is_equiv_unitary(U, V)


def ceshi_decompose(g, decomp_func):
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
    assert_equivalent_unitary(U, circ.unitary())
