import argparse

import dwave_networkx

import networkx as nx
import json

from pegasustools.util.adj import save_graph_adjacency, canonical_order_labels

from dwave.system import DWaveSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate DW graph topologies"
    )

    parser.add_argument("-L", type=int, default=8, help="System linear length scale L.")
    parser.add_argument("--labels", type=str, default="labels.json",
                        help="Save file for the integer label mapping of the QAC graph")
    # parser.add_argument("--all-qubits", action='store_true',
    #                     help="Generate a topolgy with all working qubits on the Pegasus graph. "
    #                          "Does not take the QPU into account ")
    parser.add_argument("--graphml", type=str, default=None,
                        help="Optionally, a .graphml file to store a complete specification of the graph")
    parser.add_argument("dest", type=str,
                        help="Save file for the topology in text adjacency list format")

    args = parser.parse_args()

    l = args.L
    sampler = DWaveSampler()
    g = sampler.to_networkx_graph()
    g_sub = dwave_networkx.pegasus_graph(l, node_list=g.nodes, edge_list=g.edges)

    mapping_dict, g2 = canonical_order_labels(g_sub)
    nx.set_node_attributes(g2, mapping_dict['labels_to_nodes'], "qubit")
    # Save the integer label adjacency list in plain text
    save_graph_adjacency(g2, args.dest)

    with open(args.labels, 'w') as f:
        json.dump(mapping_dict, f)

    if args.graphml is not None:
        nx.readwrite.write_graphml(g2, args.graphml)


if __name__ == "__main__":
    main()
