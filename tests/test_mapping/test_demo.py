import networkx as nx
from unisys import gate, circuit

from tests.ceshi_common import ceshi_mapping


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


############################################
# coupling graph from file
############################################
def test_oslo():
    ceshi_mapping(circ, device_fname='./topology/oslo.graphml')


############################################
# grid graph
############################################
def test_grid():
    from unisys.utils.arch import gene_grid_2d_graph
    ceshi_mapping(circ, gene_grid_2d_graph(circ.num_qubits))


############################################
# random connected graph
############################################
def test_random():
    ceshi_mapping(circ, nx.connected_watts_strogatz_graph(circ.num_qubits, 2, 0.5))


############################################
# chain graph
############################################
def test_chain():
    ceshi_mapping(circ, nx.path_graph(circ.num_qubits))
