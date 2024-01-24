from unisys import gate
from unisys import decompose

from tests.ceshi_common import ceshi_decompose


##########################################
# Pauli-related gates decomposition
##########################################
def test_X_1_02():
    ceshi_decompose(gate.X.on([1], [0, 2]), decompose.ccx_decompose)


def test_Y_1_0():
    ceshi_decompose(gate.Y.on(1, 0), decompose.cy_decompose)


def test_Z_1_0():
    ceshi_decompose(gate.Z.on(1, 0), decompose.cz_decompose)


def test_RX_2_0():
    ceshi_decompose(gate.RX(1.2).on(2, 0), decompose.crx_decompose)


def test_RY_2_0():
    ceshi_decompose(gate.RY(1.2).on(2, 0), decompose.cry_decompose)


def test_RZ_2_0():
    ceshi_decompose(gate.RZ(1.2).on(2, 0), decompose.crz_decompose)


##########################################
# S/H/T-related gates decomposition
##########################################
def test_H_1_0():
    ceshi_decompose(gate.H.on(1, 0), decompose.ch_decompose)


def test_S_1_0():
    ceshi_decompose(gate.S.on(1, 0), decompose.cs_decompose)


def test_T_1_0():
    ceshi_decompose(gate.T.on(1, 0), decompose.ct_decompose)


##########################################
# SWAP-related decomposition
##########################################
def test_swap_01():
    ceshi_decompose(gate.SWAP.on([0, 1]), decompose.swap_decompose)


def test_swap_01_2():
    ceshi_decompose(gate.SWAP.on([0, 1], [2]), decompose.cswap_decompose)
