import os
import networkx as nx
from unisys import Circuit
from unisys.utils import arch

from tests.ceshi_common import ceshi_mapping

dpath = './circuit'
fnames = os.listdir(dpath)
fnames = [os.path.join(dpath, fname) for fname in fnames]


############################################
# benchmarks on chain/path graph
############################################
def test_path_coupling_graph():
    for fname in fnames:
        circ = Circuit.from_qasm(fname=fname)
        ceshi_mapping(circ, nx.path_graph(circ.num_qubits))


############################################
# benchmarks on grid graph
############################################
def test_grid_coupling_graph():
    for fname in fnames:
        circ = Circuit.from_qasm(fname=fname)
        ceshi_mapping(circ, arch.gene_grid_2d_graph(circ.num_qubits))


############################################
# benchmarks on random graph
############################################
def test_random_connected_graph():
    for fname in fnames:
        circ = Circuit.from_qasm(fname=fname)
        ceshi_mapping(circ, nx.connected_watts_strogatz_graph(circ.num_qubits, 2, 0.5))
