"""
Universal graph operation utilities, especially for "rustworkx" backend
"""
import pydot
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image
from typing import List, Any, Callable
from networkx.drawing.nx_agraph import graphviz_layout
from unisys.basic.gate import Gate


def draw_circ_dag_mpl(dag: nx.DiGraph, fname=None, figsize=None, fix_layout=False):
    colors = {1: 'white', 2: 'lightblue', 3: 'lightgreen', 4: 'lightpink',
              5: 'lightyellow', 6: 'lightgray', 7: 'lightcyan', 8: 'lightcoral'}
    node_colors = [colors[g.num_qregs] if isinstance(g, Gate) else colors[g.num_qubits] for g in dag.nodes]
    if fix_layout:
        pos = graphviz_layout(dag, prog='dot')
    else:
        pos = None

    if figsize:
        plt.figure(figsize=figsize)

    nx.draw(dag, pos, with_labels=True,
            labels={g: (g.math_repr() if isinstance(g, Gate) else str(g)) for g in dag.nodes}, node_color=node_colors,
            edgecolors='grey',
            node_size=450, font_size=8, font_weight='bold')
    if fname:
        plt.savefig(fname)


def draw_circ_dag_graphviz(dag: nx.DiGraph, fname: str = None) -> Image:
    dot = pydot.Dot(graph_type='digraph')
    gate_to_node = {}
    colors = {1: 'white', 2: 'lightblue', 3: 'lightgreen', 4: 'lightpink',
              5: 'lightyellow', 6: 'lightgray', 7: 'lightcyan', 8: 'lightcoral'}
    for g in dag.nodes:
        node = pydot.Node(hash(g), label=str(g),
                          fillcolor=colors[g.num_qregs] if isinstance(g, Gate) else colors[g.num_qubits],
                          style='filled')
        gate_to_node[g] = node
        dot.add_node(node)
    for edge in dag.edges:
        dot.add_edge(pydot.Edge(gate_to_node[edge[0]], gate_to_node[edge[1]]))
    dot.set_rankdir('LR')
    if fname:
        dot.write_png(fname)
    return Image(dot.create_png())


def find_successors_by_node(dag: nx.DiGraph, node: Any, predicate: Callable) -> List[Any]:
    """
    Return a filtered list of successors data such that each node matches the filter.

    Args:
        dag: The DAG to search
        node: The node to get the successors for
        predicate: The filter function to use for matching each of its successor nodes

    Returns:
        A list of the node data for all the successors who match the filter
    """
    return [node for node in dag.successors(node) if predicate(node)]


def find_predecessors_by_node(dag: nx.DiGraph, node: int, predicate: Callable) -> List[Any]:
    """
    Return a filtered list of predecessors data such that each node has at least one edge data which matches the filter.

    Args:
        dag: The DAG to search
        node: The node to get the predecessors for
        predicate: The filter function to use for matching each of its predecessor nodes

    Returns:
        A list of the node data for all the predecessors who match the filter
    """
    return [node for node in dag.predecessors(node) if predicate(node)]


def filter_nodes(dag: nx.DiGraph, predicate: Callable) -> List[Any]:
    """Return a list of node indices for all nodes in a graph which match the filter."""
    return [node for node in dag.nodes() if predicate(node)]
