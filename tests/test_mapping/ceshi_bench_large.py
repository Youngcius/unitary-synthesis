# NOTE: this test sets consumes a lot of time
import os
from unisys import Circuit
from unisys.utils import arch
from tests.ceshi_common import ceshi_mapping
from rich import console
import networkx as nx

console = console.Console()

large_circ_path = '../../benchmark/circuit/large'
fnames = os.listdir(large_circ_path)
fnames = [os.path.join(large_circ_path, fname) for fname in fnames]


def test_path_coupling_graph():
    for fname in fnames:
        circ = Circuit.from_qasm(fname=fname)
        ceshi_mapping(circ, nx.path_graph(circ.num_qubits))


def test_grid_coupling_graph():
    for fname in fnames:
        circ = Circuit.from_qasm(fname=fname)
        ceshi_mapping(circ, arch.gene_grid_2d_graph(circ.num_qubits))


def test_random_connected_graph():
    for fname in fnames:
        circ = Circuit.from_qasm(fname=fname)
        ceshi_mapping(circ, nx.connected_watts_strogatz_graph(circ.num_qubits, 2, 0.5))
