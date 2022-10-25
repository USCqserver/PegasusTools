import argparse
import numpy as np
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import json
from pegasustools.qac import PegasusQACEmbedding, PegasusQACGraph
from pegasustools.nqac import PegasusK4NQACGraph
from pegasustools.util.adj import save_graph_adjacency, save_ising_instance_graph, read_ising_adjacency_graph, \
    read_mapping, canonical_order_labels

from dwave.system import DWaveSampler


def honeycomb_lattice(n, m=None, periodic=False, ladder_bonds=False):
    """
    Generate a planar honeycomb/hexagonal lattice
    :param n:
    :return:
    """
    if m is None:
        m = n//2
    node_list = []
    edge_list = []
    pos_list = {}
    d0 = 0.8660254037844387  # cos(pi/6)
    r = 1 / (2 * d0)
    drk = r * np.asarray([[0.0, 1.0], [-d0, 0.5],
                      [-d0, -0.5], [0.0, -1.0]])
    for y in range(m):
        for x in range(n):
            for k in range(4):
                node_list.append((x, y, k))
                rc = np.asarray([x, 3*r*y])
                rk = rc + drk[k]
                pos_list[(x, y, k)] = rk

            edge_list += [((x, y, i), (x, y, i+1)) for i in range(3)]
            if x < n-1 or periodic:
                edge_list += [((x, y, 0), ((x+1)%n, y, 1)),
                              ((x, y, 3), ((x+1)%n, y, 2))]
            if y < m-1 or periodic:
                edge_list += [((x, y, 0), (x, (y+1)%m, 3))]

    if ladder_bonds:
        for y in range(m):
            for x in range(n):
                if (x < n -2 and y < m-1) or periodic:
                    edge_list.append(((x, y, 0), ((x+2)%n, (y+1)%m, 2)))
                if (x < n-1) or periodic:
                    edge_list.append(((x, y, 2), ((x+1)%n, y, 0)))
                if (x < n-1) or periodic:
                    edge_list.append(((x, y, 1), ((x+1)%n, y, 3)))
                if (x < n-2 and y > 1) or periodic:
                    edge_list.append(((x, y, 3), ((x+2)%n, (y-1)%m, 1)))

    g = nx.Graph()
    g.add_nodes_from(node_list)
    g.add_edges_from(edge_list)
    nx.set_node_attributes(g, pos_list, "pos")
    return g


def main():
    parser = argparse.ArgumentParser(
        description="Generate regular graph topologies"
    )

    parser.add_argument("-L", type=int, default=8, help="System linear length scale L.")
    parser.add_argument("--labels", type=str, default="labels.json",
                        help="Save file for the integer label mapping of the QAC graph")
    parser.add_argument("--all-qubits", action='store_true',
                        help="Generate a topolgy with all working qubits on the Pegasus graph. "
                             "Does not take the QPU into account ")
    parser.add_argument("--graphml", type=str, default=None,
                        help="Optionally, a .graphml file to store a complete specification of the graph")
    parser.add_argument("--plot", type=str, default=None,
                        help="Optionally, output a plot of the graph.")
    parser.add_argument("--periodic", action='store_true')
    parser.add_argument("top", type=str,
                        help="Name of the topology to generate")
    parser.add_argument("dest", type=str,
                        help="Save file for the topology in text adjacency list format")

    args = parser.parse_args()

    l = args.L
    if args.top == "hc":
        g = honeycomb_lattice(l, periodic=args.periodic)
    elif args.top == "hc-qac":
        g = honeycomb_lattice(l, periodic=args.periodic, ladder_bonds=True)
    else:
        raise RuntimeError(f"Unrecognized topology {args.top}")
    # The integer ordering of a QAC graph is mapped from the lexicographic ordering
    # of the logical node coordinates (t, x, z, u), regardless of inactive logical qubits
    # This means that unused integer labels can be skipped
    mapping_dict, g2 = canonical_order_labels(g)
    # Set the original logical QAC node string '(t,x,z,u)' as an attribute
    nx.set_node_attributes(g2, mapping_dict['labels_to_nodes'], "qubit")
    # Save the integer label adjacency list in plain text
    save_graph_adjacency(g2, args.dest)

    with open(args.labels, 'w') as f:
        json.dump(mapping_dict, f)

    if args.graphml is not None:
        nx.readwrite.write_graphml(g2, args.graphml)

    if args.plot is not None:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        nx.draw_networkx(g, pos=nx.get_node_attributes(g, "pos"),
                         edge_color=(0.2,0.2,0.2,0.4),
                         with_labels=False, font_size=12, node_size=75)
        plt.savefig(args.plot)


if __name__ == "__main__":
    main()
