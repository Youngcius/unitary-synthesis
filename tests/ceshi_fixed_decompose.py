from unisys import gate, decompose
from ceshi_common import ceshi_decompose

if __name__ == '__main__':
    # Pauli-related gates decomposition
    print('Pauli-related gates decomposition')
    ceshi_decompose(gate.X.on([1], [0, 2]), decompose.ccx_decompose)
    ceshi_decompose(gate.Y.on(1, 0), decompose.cy_decompose)
    ceshi_decompose(gate.Z.on(1, 0), decompose.cz_decompose)
    ceshi_decompose(gate.RX(1.2).on(2, 0), decompose.crx_decompose)
    ceshi_decompose(gate.RY(1.2).on(2, 0), decompose.cry_decompose)
    ceshi_decompose(gate.RZ(1.2).on(2, 0), decompose.crz_decompose)
    print()

    # SHT-related gates decomposition
    print('SHT-related gates decomposition')
    ceshi_decompose(gate.H.on(1, 0), decompose.ch_decompose)
    ceshi_decompose(gate.S.on(1, 0), decompose.cs_decompose)
    ceshi_decompose(gate.T.on(1, 0), decompose.ct_decompose)
    print()

    # SWAP decomposition
    print('SWAP decomposition')
    ceshi_decompose(gate.SWAP.on([0, 1]), decompose.swap_decompose)

