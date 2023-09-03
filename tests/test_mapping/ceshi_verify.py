from unisys import gate, circuit
from unisys.utils import arch


def ceshi_verify():
    inserted_swap = gate.SWAP.on([0, 3])
    circ = circuit.Circuit([
        gate.H.on(0), gate.H.on(2), gate.H.on(5),
        gate.Z.on(0), gate.X.on(2, 1), gate.X.on(5, 4),
        gate.X.on(1, 0), gate.X.on(3, 2),
        gate.H.on(2), gate.H.on(3),
        gate.X.on(2, 1), gate.X.on(5, 3),
        gate.Z.on(3),
        gate.X.on(3, 4),
        gate.X.on(0, 3)
    ])
    print(circ)
    print(circ.to_qiskit().draw())

    circ_with_swaps = circ.clone()
    circ_with_swaps.insert(-2, inserted_swap)

    device = arch.read_device_topology('../../benchmark/topology/oslo.graphml')
    mapping = arch.gene_init_mapping(circ, device)
    updated_mapping = arch.update_mapping(mapping, inserted_swap)
    print('Initial mapping:', mapping)
    print('Updated mapping: {} after inserting {}'.format(updated_mapping, inserted_swap))

    mapped_circ = arch.unify_mapped_circuit(circ_with_swaps, [mapping, updated_mapping])
    print(mapped_circ)
    print(mapped_circ.to_qiskit().draw())

    assert arch.verify_mapped_circuit(circ, mapped_circ, mapping, updated_mapping)


def ceshi_verify_by_state():
    inserted_swap = gate.SWAP.on([0, 3])
    circ = circuit.Circuit([
        gate.H.on(0), gate.H.on(2), gate.H.on(5),
        gate.Z.on(0), gate.X.on(2, 1), gate.X.on(5, 4),
        gate.X.on(1, 0), gate.X.on(3, 2),
        gate.H.on(2), gate.H.on(3),
        gate.X.on(2, 1), gate.X.on(5, 3),
        gate.Z.on(3),
        gate.X.on(3, 4),
        gate.X.on(0, 3)
    ])
    print(circ)
    print(circ.to_qiskit().draw())

    circ_with_swaps = circ.clone()
    circ_with_swaps.insert(-2, inserted_swap)

    device = arch.read_device_topology('../benchmark/topology/oslo.graphml')
    mapping = arch.gene_init_mapping(circ, device)
    updated_mapping = arch.update_mapping(mapping, inserted_swap)
    print('Initial mapping:', mapping)
    print('Updated mapping: {} after inserting {}'.format(updated_mapping, inserted_swap))

    mapped_circ = arch.unify_mapped_circuit(circ_with_swaps, [mapping, updated_mapping])
    print(mapped_circ)
    print(mapped_circ.to_qiskit().draw())

    assert arch.verify_mapped_circuit_by_state(circ, mapped_circ, mapping, updated_mapping)


ceshi_verify()
