from unisys import circuit, gate
from unisys import transform, partition
import cirq


def test_unroll_su4_to_cnot():
    """Partition --> Fuse --> Unroll --> Check"""
    circ = circuit.Circuit([
        gate.H.on(0), gate.H.on(2), gate.H.on(5),
        gate.Z.on(0),
        gate.X.on(2, 1), gate.X.on(5, 4),
        gate.X.on(1, 0), gate.X.on(3, 2),
        gate.H.on(2), gate.H.on(3),
        gate.X.on(2, 1), gate.X.on(5, 3),
        gate.Z.on(3),
        gate.X.on(3, 4),
        gate.X.on(0, 3)
    ])

    blocks = partition.greedy_partition(circ, grain=2)
    fused = transform.fuser.fuse_blocks(blocks)
    circ_unrolled = transform.unroller.unroll_su4(fused, by='cnot')
    cirq.testing.assert_allclose_up_to_global_phase(
        circ_unrolled.unitary(),
        circ.unitary(),
        atol=1e-7
    )


def test_unroll_su4_to_can():
    """Partition --> Fuse --> Unroll --> Check"""
    circ = circuit.Circuit([
        gate.H.on(0), gate.H.on(2), gate.H.on(5),
        gate.Z.on(0),
        gate.X.on(2, 1), gate.X.on(5, 4),
        gate.X.on(1, 0), gate.X.on(3, 2),
        gate.H.on(2), gate.H.on(3),
        gate.X.on(2, 1), gate.X.on(5, 3),
        gate.Z.on(3),
        gate.X.on(3, 4),
        gate.X.on(0, 3)
    ])

    blocks = partition.greedy_partition(circ, grain=2)
    fused = transform.fuser.fuse_blocks(blocks)
    circ_unrolled = transform.unroller.unroll_su4(fused, by='can')
    cirq.testing.assert_allclose_up_to_global_phase(
        circ_unrolled.unitary(),
        circ.unitary(),
        atol=1e-7
    )
