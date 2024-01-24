import networkx as nx
from unisys.utils import arch

from tests.ceshi_common import ceshi_mapping

circ = arch.gene_random_circuit(8, 100)


############################################
# random circuit test on grid graph
############################################
def test_grid():
    print('test random circuit on grid graph')
    ceshi_mapping(circ, arch.gene_grid_2d_graph(circ.num_qubits))


############################################
# random circuit on random connected graph
############################################
def test_random():
    print('test random circuit on random graph')
    ceshi_mapping(circ, nx.connected_watts_strogatz_graph(circ.num_qubits, 2, 0.5))


############################################
# random circuit on chain/path graph
############################################
def test_chain():
    print('test random circuit on chain graph')
    ceshi_mapping(circ, nx.path_graph(circ.num_qubits))


############################################
# random circuit test on IBM Oslo
############################################
def test_oslo():
    print('test random circuit mapping on IBM Oslo')
    circ = arch.gene_random_circuit(6, 100)
    ceshi_mapping(circ, arch.read_device_topology('./topology/oslo.graphml'))
