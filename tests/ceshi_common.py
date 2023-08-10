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


def ceshi_mapping(circ, device=None, device_fname=None):
    if device is None and device_fname is None:
        raise ValueError('Either device or device_fname should be specified')
    if device is None:
        from unisys.utils.arch import read_device_topology
        device = read_device_topology(device_fname)
    from unisys.mapping.heuristic import sabre_search
    from unisys.utils.arch import verify_mapped_circuit
    print(device)
    print(device.nodes)
    print(device.edges)

    mapped_circ, init_mapping, final_mapping = sabre_search(circ, device)
    print('Initial mapping:', init_mapping)
    print('Final mapping:\t', final_mapping)
    print('Original circuit:', circ)
    print(circ.to_cirq())
    print('Mapped circuit:', mapped_circ)
    print(mapped_circ.to_cirq())
    assert verify_mapped_circuit(circ, mapped_circ, init_mapping, final_mapping)
    print('Mapping correct!')
