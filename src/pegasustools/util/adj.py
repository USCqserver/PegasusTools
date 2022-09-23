from typing import List, Dict
from dimod import BinaryQuadraticModel, AdjVectorBQM
#AdjacencyList = List[ Dict[int, float]]
import networkx as nx
import json
from ast import literal_eval


def canonical_order_labels(g: nx.Graph, to_str=True):
    sorted_nodes = sorted(g.nodes())
    node_labels = {n: i for i, n in enumerate(sorted_nodes)}

    g2 = nx.Graph()
    g2.add_nodes_from(g)
    g2.add_edges_from(g.edges)
    g2 = nx.relabel_nodes(g2, node_labels)
    if to_str:
        str_node_labels = {str(n): i for i, n in enumerate(sorted_nodes)}
        label_nodes = {i: str(n) for i, n in enumerate(sorted_nodes)}
        mapping_dict = {"nodes_to_labels": str_node_labels, "labels_to_nodes": label_nodes}
    else:
        label_nodes = {i: n for i, n in enumerate(sorted_nodes)}
        mapping_dict = {"nodes_to_labels": node_labels, "labels_to_nodes": label_nodes}
    return mapping_dict, g2


def read_problem_adjacency(filename, max_k=1.0, sep=None, qubo=False):
    """
        Reads a three-column text file specifying the adjacency
        Returns the BQM in the respective ising or qubo format

        :param filename:
        :return:
        """
    linear = {}
    quadratic = {}

    with open(filename) as f:
        for l, line in enumerate(f):
            toks = line.split(sep=sep)
            if len(toks) != 3:
                raise ValueError(f"Expected three tokens in line {l}")
            i, j, K = int(toks[0]), int(toks[1]), float(toks[2])
            if qubo:
                (i2, j2) = (i, j) if i < j else (j, i)
                quadratic[(i2, j2)] = K / max_k
            else:
                if i == j:
                    linear[i] = K / max_k
                else:
                    (i2, j2) = (i, j) if i < j else (j, i)
                    quadratic[(i2, j2)] = K / max_k

    if qubo:
        bqm = AdjVectorBQM.from_qubo(quadratic)
    else:
        bqm = AdjVectorBQM.from_ising(linear, quadratic)
    return bqm


def read_ising_adjacency(filename, max_k=1.0, sep=None, qubo=False):
    """
    Reads a three-column text file specifying the adjacency

    If the adjacency is in qubo format, it is automatically transformed to Ising format
    via the transformation b = (1 + s)/2 (dimod convention)
    i.e. b=0  -> s=-1
         b=1  -> s=+1
    :param filename:
    :return:
    """
    bqm = read_problem_adjacency(filename, max_k, sep=sep, qubo=qubo)

    if qubo:
        bqm = AdjVectorBQM.from_ising(*bqm.to_ising())

    return bqm


def read_ising_adjacency_graph(filename, max_k=1.0):
    """
    Reads a three-column text file specifying the problem adjacency
    Self-loops are interpreted as Ising biases
    :param filename:
    :return: Undirected NetworkX graph of the problem
    """
    g = nx.Graph()

    with open(filename) as f:
        for l, line in enumerate(f):
            toks = line.split()
            if len(toks) != 3:
                raise ValueError(f"Expected three tokens in line {l}")
            i, j, K = int(toks[0]), int(toks[1]), float(toks[2])
            if i == j:
                g.add_node(i, bias=K/max_k)
            else:
                (i2, j2) = (i, j) if i < j else (j, i)
                g.add_edge(i2, j2, weight= K/max_k)

    return g


def save_ising_instance_graph(g: nx.Graph, filename):
    with open(filename, 'w') as f:
        for n, h in g.nodes.data("bias"):
            if h is not None:
                f.write(f"{n} {n} {h}\n")
        for u, v, j in g.edges.data("weight"):
            if j is not None:
                f.write(f"{u} {v} {j}\n")


def ising_graph_to_bqm(g: nx.Graph):
    linear = {}
    quadratic = {}
    for n, h in g.nodes.data("bias"):
        if h is not None:
            linear[n] = h
    for u, v, j in g.edges.data("weight"):
        if j is not None:
            (u2, v2) = (u, v) if u < v else (v, u)
            quadratic[(u2, v2)] = j

    bqm = AdjVectorBQM.from_ising(linear, quadratic)
    return bqm

def save_graph_adjacency(g: nx.Graph, filename):
    nx.readwrite.write_adjlist(g, filename)


def read_mapping(filename):
    """
    Reads a mapping file
    :param filename:
    :return:  nodes_to_labels, labels_to_nodes
    """
    with open(filename, 'r') as f:
        mapping_dict = json.load(f)

    nodes_to_labels = mapping_dict["nodes_to_labels"]
    labels_to_nodes = mapping_dict["labels_to_nodes"]
    nodes_to_labels = {literal_eval(n): i for n, i in nodes_to_labels.items()}
    labels_to_nodes = {literal_eval(i): literal_eval(n) for i, n in labels_to_nodes.items()}

    return nodes_to_labels, labels_to_nodes
