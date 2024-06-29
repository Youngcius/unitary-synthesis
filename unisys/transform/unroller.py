"""
Unroll some specific-type gates in quantum circuits
"""
from unisys.basic.circuit import Circuit
from unisys.basic import gate
from unisys.utils.operations import is_tensor_prod
from unisys import decompose


def unroll_u3(circ: Circuit, by: str = 'zyz') -> Circuit:
    """
    Unroll U1, U2 and U3 gate by Euler decomposition
    """
    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == 1 and g.name in ['U2', 'U3']:
            circ_unrolled += decompose.euler_decompose(g, basis=by, with_phase=False)
        elif g.num_qregs == 1 and g.name == 'U1':
            circ_unrolled.append(gate.RZ(g.angle).on(g.tq))
        else:
            circ_unrolled.append(g)

    return circ_unrolled


def unroll_su4(circ: Circuit, by: str = 'can') -> Circuit:
    """
    Unroll two-qubit gates, i.e., SU(4) gates, by canonical decomposition or other methods
    """
    if by not in ['can', 'cnot']:
        raise ValueError("Only support canonical (by='can') and CNOT unrolling (by='cnot').")
    
    circ_unrolled = Circuit()
    for g in circ:
        if g.num_qregs == 2 and not g.cqs:  # arbitrary two-qubit gate
            if by == 'can':
                circ_unrolled += decompose.can_decompose(g)
            if by == 'cnot':
                circ_unrolled += decompose.kak_decompose(g)
        else:
            circ_unrolled.append(g)

    return circ_unrolled


def unroll_to_su4(circ: Circuit, method: str = 'approx') -> Circuit:
    """
    Unroll multi-qubit gates into SU(4) gates by exact (algorithmic) or approximate (numeric) methods
    """
    raise NotImplementedError


def unroll_tensor_product(circ: Circuit) -> Circuit:
    """
    Unroll fake two-qubit gate (tensor-product) into two singel-qubit gates
    """
    circ_unrolled = Circuit()
    for g in circ:
        if is_tensor_prod(g.data):
            circ_unrolled += decompose.tensor_product_decompose(g)
        else:
            circ_unrolled.append(g)

    return circ_unrolled
