import argparse
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import json
from pegasustools.qac import PegasusQACEmbedding, PegasusQACGraph
from pegasustools.util.adj import save_graph_adjacency, save_ising_instance_graph, read_ising_adjacency_graph

from dwave.system import DWaveSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph data of the native QAC topology"
    )

    parser.add_argument("-L", type=int, default=None, help="System size L. If none, uses all available QAC nodes")
    parser.add_argument("--labels", type=str, default="labels.json",
                        help="Save file for the integer label mapping of the QAC graph")
    parser.add_argument("--all-qubits", action='store_true',
                        help="Generate a topolgy with all working qubits on the Pegasus graph. "
                             "Does not take the QPU into account ")
    parser.add_argument("--graphml", type=str, default=None,
                        help="Optionally, a .graphml file to store a complete specification of the graph")
    parser.add_argument("--plot", type=str, default=None,
                        help="Optionally, output a PNG plot of the graph.")
    parser.add_argument("--plot-instance", type=str, default=None)
    parser.add_argument("--percolation", type=str, default=None,
                        help="Optionally, write a percolation instance to this file."
                        "All couplings are ferromagnetic, the top qubits of the graph are marked with a +1 bias "
                        "and the bottom qubits are marked with a -1 bias")
    parser.add_argument("dest", type=str,
                        help="Save file for the QAC topology in text adjacency list format")

    args = parser.parse_args()

    if args.all_qubits:
        dw_graph: nx.Graph = dnx.pegasus_graph(16)
        dw_nodes = set(dw_graph.nodes)
        dw_edges = set(dw_graph.edges)
        qac_graph = PegasusQACGraph(16, dw_nodes, dw_edges)
    else:
        dw_sampler = DWaveSampler()
        qac_sampler = PegasusQACEmbedding(16, dw_sampler)
        qac_graph = qac_sampler.qac_graph

    if args.L is not None:
        l = args.L
        qac_graph = qac_graph.subtopol(args.L)
    else:
        l = 15
    g = qac_graph.g.copy()

    # create a percolation instance
    if args.percolation is not None:
        for (e, ed) in g.edges.items():
            ed["weight"] = -1.0
        for (n, nd) in g.nodes.items():
            if n[1] == 0:
                nd["bias"] = 1.0
            elif n[1] == l-1:
                nd["bias"] = -1.0

    # The integer ordering of a QAC graph is mapped from the lexicographic ordering
    # of the logical node coordinates (t, x, z, u), regardless of inactive logical qubits
    # This means that unused integer labels can be skipped
    sorted_nodes = sorted(g.nodes())
    node_labels = {n: i for i, n in enumerate(sorted_nodes)}
    str_node_labels = {str(n): i for i, n in enumerate(sorted_nodes)}
    label_nodes = {i: str(n) for i, n in enumerate(sorted_nodes)}
    mapping_dict = {"nodes_to_labels": str_node_labels, "labels_to_nodes": label_nodes}
    # Relabel the graph to integer labels
    g2 = nx.relabel_nodes(g, node_labels, copy=True)
    # Set the original logical QAC node string '(t,x,z,u)' as an attribute
    nx.set_node_attributes(g2, label_nodes, "qubit")
    # Save the integer label adjacency list in plain text
    save_graph_adjacency(g2, args.dest)
    if args.plot is not None:
        fig, ax = plt.subplots(figsize=(12, 12))
        if args.plot_instance is not None:
            instance = read_ising_adjacency_graph(args.plot_instance)
            cols = []
            for u, v in g2.edges:
                if instance.has_edge(u, v):
                    if 'weight' in instance.edges[u,v]:
                        cols.append(instance.edges[u, v]['weight'])
                    else:
                        cols.append(0.0)
                else:
                    cols.append(0.0)
            qac_graph.draw(edge_cmap=plt.cm.bwr, node_size=25, alpha=0.6, width=2.0,
                           edge_color=cols, edge_vmin=-3.0, edge_vmax=3.0)
        else:
            qac_graph.draw(node_size=75,)
        plt.savefig(args.plot)
    if args.percolation is not None:
        save_ising_instance_graph(g2, args.percolation)

    if args.graphml is not None:
        # stringify qubit locations, then save graphml
        #for n in g2.nodes:
        #    g2.nodes[n]["qubit"] = str(g2.nodes[n]["qubit"])
        nx.readwrite.write_graphml(g2, args.graphml)
    with open(args.labels, 'w') as f:
        json.dump(mapping_dict, f)


if __name__ == "__main__":
    main()
