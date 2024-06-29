import networkx as nx
from unisys import gate, circuit
from unisys.utils import arch
from tests.ceshi_common import ceshi_mapping
from unisys.mapping import heuristic
from rich import console

console = console.Console()

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
# ceshi_mapping(circ, device_fname='../../benchmark/topology/oslo.graphml')
device = arch.read_device_topology('./topology/oslo.graphml')
console.print(device.edges)
# mapped_circ, init_mapping, final_mapping = heuristic.sabre_search(circ, device)

mapped_circ, init_mapping, final_mapping = heuristic.sabre_search(circ, device)

console.rule('FINAL RESULT')
console.print('origin circuit: {}'.format(circ))
print(circ.to_cirq())
console.print('mapped circuit: {}'.format(mapped_circ))
print(mapped_circ.to_cirq())
console.print('initial mapping: {}'.format(init_mapping))
console.print('final mapping: {}'.format(final_mapping))
print(arch.verify_mapped_circuit(circ, mapped_circ, init_mapping, final_mapping))

############################################
# grid graph
############################################
# ceshi_mapping(circ, arch.gene_grid_2d_graph(circ.num_qubits))

############################################
# random connected graph
############################################
# ceshi_mapping(circ, nx.connected_watts_strogatz_graph(circ.num_qubits, 2, 0.5))

############################################
# chain graph
############################################
# ceshi_mapping(circ, nx.path_graph(circ.num_qubits))
