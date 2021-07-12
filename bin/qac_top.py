import argparse
import networkx as nx
import json
from pegasustools.qac import PegasusQACEmbedding
from pegasustools.util.adj import save_graph_adjacency

from dwave.system import DWaveSampler


parser = argparse.ArgumentParser(
    description="Generate graph data of the native QAC topology"
)

parser.add_argument("-L", type=int, default=None, help="System size L. If none, uses all available QAC nodes")
parser.add_argument("--labels", type=str, default="labels.json",
                    help="Save file for the integer label mapping of the QAC graph")
parser.add_argument("dest", type=str,
                    help="Save file for the QAC topology in text adjacency list format")

args = parser.parse_args()


dw_sampler = DWaveSampler()
qac_sampler = PegasusQACEmbedding(16, dw_sampler)
if args.L is not None:
    qac_graph = qac_sampler.qac_graph.subtopol(args.L)
else:
    qac_graph = qac_sampler.qac_graph
g = qac_graph.g
# The integer ordering of a QAC graph is mapped from the lexicographic ordering
# of the logical node coordinates (t, x, z, u)
sorted_nodes = sorted(g.nodes())
node_labels = {str(n): i for i, n in enumerate(sorted_nodes)}
label_nodes = {i: str(n) for i, n in enumerate(sorted_nodes)}
mapping_dict = {"nodes_to_labels": node_labels, "labels_to_nodes": label_nodes}

g2 = nx.convert_node_labels_to_integers(g, ordering="sorted", label_attribute="qubit")
save_graph_adjacency(g2, args.dest)
with open(args.labels, 'w') as f:
    json.dump(mapping_dict, f)
