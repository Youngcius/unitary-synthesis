import networkx as nx
from unisys.utils import arch
from tests.ceshi_common import ceshi_mapping

############################################
# random circuit test on IBM Oslo
############################################
circ = arch.gene_random_circuit(6, 100)
ceshi_mapping(circ, arch.read_device_topology('../../benchmark/topology/oslo.graphml'))

############################################
# random circuit test on grid graph
############################################
num_qubits = 8
depth = 100
circ = arch.gene_random_circuit(num_qubits, depth)
ceshi_mapping(circ, arch.gene_grid_2d_graph(circ.num_qubits))

############################################
# random circuit on random connected graph
############################################
ceshi_mapping(circ, nx.connected_watts_strogatz_graph(circ.num_qubits, 2, 0.5))
