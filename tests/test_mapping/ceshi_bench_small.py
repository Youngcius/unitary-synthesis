"""
These benchmarks are from https://github.com/vishal929/qubitMapper, 
which are originally from https://github.com/CQCL/tket_benchmarking and other benchmarking suites.
"""
import os
from unisys import Circuit
from unisys.utils import arch
from tests.ceshi_common import ceshi_mapping
from rich import console
import networkx as nx

console = console.Console()

small_circ_path = '../../benchmarks/small'
fnames = os.listdir(small_circ_path)
fnames = [os.path.join(small_circ_path, fname) for fname in fnames]


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
