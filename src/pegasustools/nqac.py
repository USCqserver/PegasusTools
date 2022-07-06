import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, TypeVar, Set, Iterable
from itertools import combinations, product
from pegasustools.qac import purge_deg1, AbstractQACGraph
from pegasustools.pqubit import collect_available_unit_cells, Pqubit
from pegasustools.util.graph import random_walk_loop

LQ = TypeVar('LQ')
PQ = TypeVar('PQ')


def lookup_all_edges(node_list, edge_set):
    """
    Given a sorted list of nodes, look up all edges between them in the edge set
    Args:
        node_list:
        edge_set:

    Returns:

    """
    return [(qi, qj) for (qi, qj) in combinations(node_list, 2) if (qi, qj) in edge_set]


def lookup_bipartite_edges(node_list1, node_list2, edge_set, cmp = lambda x,y: x<y):
    """
    Given a list of nodes, look up all edges between them in the edge set
    Args:
        node_list1:
        node_list2:
        edge_set:
        cmp:
    Returns:

    """
    edges = []
    for (qi, qj) in product(node_list1, node_list2):
        if not cmp(qi, qj):
            qi, qj = qj, qi
        if (qi, qj) in edge_set:
            edges.append((qi, qj))
    return edges


def embed_logical_qubits(logical_qubits: Dict[LQ, Iterable[PQ]],
                         logical_couplings: Iterable[Tuple[LQ, LQ]],
                         edge_set: Set[Tuple[PQ, PQ]],
                         accept_qubit_crit=lambda q, lc: len(lc) > 0,
                         accept_coupler_crit=lambda q1, q2, lc: len(lc) > 0,
                         strict=False
                         ):
    """
    Nodes must be represented by objects with an order (e.g. integers) and all edges (i, j)
    should be ordered as  i < j.
    Note: we don't check yet if the graph of a physical qubit is connected.
    Args:

        logical_qubits: dictionary of logical qubits represented by the physical qubit
        logical_couplings: list of couplings between logical qubits
        edge_set: *set* of edges in the physical graph
        strict: Require all listed logical qubits and couplings to exist with at least one coupler,
            otherwise an exception is raised.
        accept_coupler_crit:
        accept_qubit_crit:
    Returns:

    """
    lq_intra_couplers = {}
    for lq, pqs in logical_qubits.items():
        pq_couplers = lookup_all_edges(pqs, edge_set)
        if accept_qubit_crit(pqs, pq_couplers):
            lq_intra_couplers[lq] = pq_couplers
        else:
            if strict:
                raise ValueError(f"Could not find valid physical embedding for logical qubit {lq} : {pqs}")

    lq_inter_couplers = {}
    for (lq1, lq2) in logical_couplings:
        if lq1 > lq2:
            lq1, lq2 = lq2, lq1
        if lq1 in lq_intra_couplers and lq2 in lq_intra_couplers:
            pqs1 = logical_qubits[lq1]
            pqs2 = logical_qubits[lq2]
            couplers_12 = lookup_bipartite_edges(pqs1, pqs2, edge_set)
            if accept_coupler_crit(pqs1, pqs2, couplers_12):
                lq_inter_couplers[(lq1, lq2)] = np.asarray(couplers_12)
            else:
                if strict:
                    raise ValueError(
                        "Could not find vaild physical coupling between logical qubits "
                        f" {lq1} : {pqs1}  and  {lq2} : {pqs2}")

    return lq_intra_couplers, lq_inter_couplers


def k4_nqac_nice2xy(t, x, z, u, a=1.0, x0=0.0, y0=0.0):
    """
    Get plottable cartesian coordinates of the cluster coordinate
    """
    # Evaluate the raw grid XY coordinates from vertical coordinates
    w = x + int(t == 0)
    # Integer image coordinates
    xv = 3 * w + t
    yv = 2 + 3*z + (2 * t) % 3

    s = 2*u - 1
    # Image coordinates to plot coordinates
    x = (float(xv) + s*0.4)
    y = -(float(yv) + s*0.35)

    return x0 + a*x, y0 + a*y


class PegasusK4NQACGraph(AbstractQACGraph):
    def __init__(self, m, node_list, edge_list, strict=True, purge_deg1=True):
        """
         Level 1 Nested QAC graph with K4 embedding
         :param m:
         :param edge_list:
         :return:
         """
        super(PegasusK4NQACGraph, self).__init__()
        edge_set = set(edge_list)

        qac_qubits = np.full((3, m - 1, m - 1, 2, 4), -1, dtype=int)
        logical_qubit_map = {}
        # iterate over native vertical coordinates
        for t in range(3):
            k = 4 * t
            for x in range(m - 1):
                w = x + int(t == 0)
                for z in range(m - 1):
                    #if not (z == 0 and t == 0):  # Not valid for the top perimeter of cells
                    qv_0 = Pqubit(m, 0, w, k, z)  # First vertical qubit in this cell
                    cell_qs = qv_0.k44_indices()  # get the list of u=0 and u=1 indices in this cell
                    qv = cell_qs[:4]
                    qh = cell_qs[4:]
                    # Specify the logical to physical qubit mapping
                    lq0 = qv[:2] + qh[:2]
                    lq1 = qv[2:] + qh[2:]
                    qac_qubits[t, x, z, 0, :] = lq0
                    qac_qubits[t, x, z, 1, :] = lq1
                    logical_qubit_map[(t, x, z, 0)] = lq0
                    logical_qubit_map[(t, x, z, 1)] = lq1

        qac_edges = set()
        # Using the nice coordinates from above (logical qubit u=0 contains the first q=0 physical qubit)
        # Furthermore, all connected node pairs are ordered lexicographically
        # Horizontal-directed external bonds
        #  (t, x, z, u)  -- (t, x+1, z, u),  0 <= x < m-2
        for t in range(3):
            for x in range(m - 2):
                for z in range(m - 1):
                    for u in range(2):
                            qac_edges.add(((t, x, z, u), (t, x + 1, z, u)))
        # Vertical-directed external bonds
        #  (t, x, z, u)  -- (t, x, z+1, u),  0 <= z < m-2 except (t=0,z=0)
        for t in range(3):
            for x in range(m - 1):
                for z in range(m - 2):
                    #if not (t == 0 and z == 0):
                    for u in range(2):
                        qac_edges.add(((t, x, z, u), (t, x, z + 1, u)))

        # Cluster bonds
        #  (t, x, z, 0)  -- (t, x, z, 1),
        for t in range(3):
            for x in range(m - 1):
                for z in range(m - 1):
                    qac_edges.add(((t, x, z, 0), (t, x, z, 1)))

        # Diagonal internal bonds
        #  (0, x, z, u1)  -- (2, x, z, u2)
        #  (2, x, z, u1)  -- (1, x, z, u2)
        #  (1, x, z, u1)  -- (0, x-1, z+1, u2)
        for x in range(m - 1):
            for z in range(m - 1):
                for u1 in range(2):
                    for u2 in range(2):
                        qac_edges.add(((0, x, z, u1), (2, x, z, u2)))
                        qac_edges.add(((1, x, z, u1), (2, x, z, u2)))
        for x in range(1, m - 1):
            for z in range(m - 2):
                for u1 in range(2):
                    for u2 in range(2):
                        qac_edges.add(((0, x - 1, z + 1, u1), (1, x, z, u2)))
        # Cross-diagonal bonds
        #  (0, x, z, u1) -- (2, x+1, z, 0)
        #  (0, x, z, 1)  -- (1, x+1, z, u2)
        #  (2, x, z, u1) -- (1, x+1, z, 0)
        #  (2, x, z, 1)  -- (0, x, z+1, u2)
        #  (1, x, z, u1) -- (0, x, z+1, 0)
        #  (1, x, z, 1)  -- (2, x, z+1, u2)
        for x in range(m - 2):
            for z in range(m - 1):
                for u in range(2):
                    qac_edges.add(((0, x, z, u), (2, x + 1, z, 0)))
                    qac_edges.add(((0, x, z, 1), (1, x + 1, z, u)))
        for x in range(m - 1):
            for z in range(m - 2):
                for u in range(2):
                    qac_edges.add(((1, x+1, z, 0), (2, x, z, u)))
                    qac_edges.add(((0, x, z+1, u), (2, x, z, 1)))
                    qac_edges.add(((0, x, z+1, 0), (1, x, z, u)))
                    qac_edges.add(((1, x, z, 1), (2, x, z+1, u)))

        # Determine the physical couplers required to embed the logical graph
        # A subgraph of K4 with at least 3 nodes and 3 edges must be connected, so we accept with this criterion
        lq_intra_couplers, lq_inter_couplers = embed_logical_qubits(
            logical_qubit_map, qac_edges, edge_set, strict=strict,
            accept_qubit_crit=lambda q, lc: len(q) >= 3 and len(lc) >= 3)

        # self.node_embeddings = lq_intra_couplers
        # self.edge_embeddings = lq_inter_couplers
        logical_nodes = set(lq_intra_couplers.keys())
        logical_edges = set(lq_inter_couplers.keys())
        self.nodes = logical_nodes
        self.edges = logical_edges
        self.qubit_array = qac_qubits
        # Make the networkx graph
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        for n in self.nodes:
            g.nodes[n]['embedding'] = lq_intra_couplers[n]
        for e in self.edges:
            g.edges[e]['embedding'] = lq_inter_couplers[e]
        self.g = g
        if purge_deg1:
            self.purge_deg1()

    def draw(self, **draw_kwargs):
        pos_list = {node: k4_nqac_nice2xy(*node, a=60.0) for node in self.nodes}
        g = self.g
        nx.draw_networkx(g, pos=pos_list, with_labels=False, font_size=12, **draw_kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dwave.system import DWaveSampler

    dws = DWaveSampler()
    qac_graph = PegasusK4NQACGraph(16, dws.nodelist, dws.edgelist, strict=False)
    fig, ax = plt.subplots(figsize=(12, 12))
    l = 4
    sub_qac = qac_graph.subtopol(l)
    n = np.random.randint(0, sub_qac.g.number_of_nodes())
    init_node = list(qac_graph.g.nodes)[n]
    rand_loop = random_walk_loop(init_node, qac_graph.g)
    print(rand_loop)

    cols = []
    edgelist = []
    for u, v, em in sub_qac.g.edges.data('embedding'):
        w = len(em)
        cols.append(w)
        edgelist.append((u, v))
    nodecols = []
    for u, em in sub_qac.g.nodes.data('embedding'):
        w = len(em)
        nodecols.append(w)
    sub_qac.draw(edge_cmap=plt.cm.get_cmap('coolwarm_r'), node_size=25, alpha=0.8, width=1.2,
                 edge_color=cols, node_color=nodecols, cmap=plt.cm.get_cmap('bwr_r'), vmin=3, vmax=6,
                 edgelist=edgelist, edge_vmin=-3.0, edge_vmax=8.0)
    # Evaluate some statistics

    #plt.show()
    plt.savefig(f"pegasus_k4_nqac_logical_l{l}.pdf")
