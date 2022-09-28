import numpy as np
import networkx as nx
import dimod
from dwave.preprocessing import ClipComposite, SpinReversalTransformComposite
from dimod import bqm_structured
from dimod import StructureComposite, Structured, bqm_structured, AdjVectorBQM
from dimod import BQM
from typing import Dict, List, Tuple, TypeVar, Set, Iterable
from itertools import combinations, product
from pegasustools.qac import purge_deg1, AbstractQACGraph, AbstractQACEmbedding
from pegasustools.pqubit import collect_available_unit_cells, Pqubit
from pegasustools.util.graph import random_walk_loop
LQ = TypeVar('LQ')
PQ = TypeVar('PQ')
PJ = Tuple[PQ, PQ]


def lookup_all_edges(node_list, edge_set):
    """
    Given a sorted list of nodes, look up all edges between them in the edge set
    Args:
        node_list:
        edge_set:

    Returns:

    """
    return [(qi, qj) for (qi, qj) in combinations(node_list, 2) if (qi, qj) in edge_set]


def lookup_bipartite_edges(node_list1, node_list2, edge_set, direct_coupling=False, cmp = lambda x,y: x<y,
                           max_edges=2):
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
    if direct_coupling:
        for (qi, qj) in zip(node_list1, node_list2):
            if not cmp(qi, qj):
                qi, qj = qj, qi
            if (qi, qj) in edge_set:
                edges.append((qi, qj))
        if len(edges) == 0:  # attempt lower register to upper register couplings
            n = len(node_list2)
            for (qi, qj) in zip(node_list1[:n//2], node_list2[n//2:]):
                if not cmp(qi, qj):
                    qi, qj = qj, qi
                if (qi, qj) in edge_set:
                    edges.append((qi, qj))
        if len(edges) == 0:  # or upper to lower register
            n = len(node_list2)
            for (qi, qj) in zip(node_list1[n//2:], node_list2[:n//2]):
                if not cmp(qi, qj):
                    qi, qj = qj, qi
                if (qi, qj) in edge_set:
                    edges.append((qi, qj))
        if len(edges) > max_edges:
            edges = (edges[::2]+edges[1::2])[:max_edges]
    else:
        for (qi, qj) in product(node_list1, node_list2):
            if not cmp(qi, qj):
                qi, qj = qj, qi
            if (qi, qj) in edge_set:
                edges.append((qi, qj))
    return edges


def embed_logical_qubits(logical_qubits: Dict[LQ, Iterable[PQ]],
                         logical_couplings: Iterable[Tuple[LQ, LQ]],
                         edge_set: Set[PJ],
                         accept_qubit_crit=lambda q, lc: len(lc) > 0,
                         accept_coupler_crit=lambda q1, q2, lc: len(lc) > 0,
                         direct_coupling=False, max_edges=2,
                         strict=False
                         ) -> Tuple[Dict[LQ, Iterable[PJ]],
                                    Dict[Tuple[LQ, LQ], Iterable[PJ]]]:
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
        n = len(pqs)
        if direct_coupling:  # Look for independent copies of K2 chains only
            pq_couplers = lookup_bipartite_edges(pqs[:n//2], pqs[n//2:], edge_set, direct_coupling=True, max_edges=n//2)
        else:  # Look for all interations among these qubits
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
            if direct_coupling:
                couplers_12 = lookup_bipartite_edges(pqs1[::2], pqs2[::2], edge_set, direct_coupling=False)
                couplers_12 += lookup_bipartite_edges(pqs1[1::2], pqs2[1::2], edge_set, direct_coupling=False)
            else:
                couplers_12 = lookup_bipartite_edges(pqs1, pqs2, edge_set)
            if accept_coupler_crit(pqs1, pqs2, couplers_12):
                lq_inter_couplers[(lq1, lq2)] = np.asarray(couplers_12)
            else:
                if strict:
                    raise ValueError(
                        "Could not find vaild physical coupling between logical qubits "
                        f" {lq1} : {pqs1}  and  {lq2} : {pqs2}")

    return lq_intra_couplers, lq_inter_couplers


def try_embed_nqac_graph(lin, qua,  # qac_map,
                        logical_qubits: Dict[LQ, Iterable[PQ]],
                        lq_intra_couplers: Dict[LQ, Iterable[PJ]],
                        lq_inter_couplers: Dict[Tuple[LQ, LQ], Iterable[PJ]],
                        avail_nodes, avail_edges, penalty_strength,
                        problem_scale=1.0, strict=True):
    """
    Linear and quadratic specifications should be Ising-type

    :param lin:
    :param qua:
    :param qac_map: Maps variables to lists of four qubits
    :param penalty_strength:
    :param problem_scale:
    :param strict:
    :return:
    """
    qac_lin = {}
    qac_qua = {}
    # Create the penalty Hamiltonian for all available QAC qubits
    for v in lin:
        q_coupl = lq_intra_couplers[v]
        for (qi, qj) in q_coupl:
            e = (qi, qj) if qi < qj else (qj, qi)
            if e not in avail_edges:
                raise RuntimeError(f"Cannot place penalty coupling {e}")
            qac_qua[e] = -penalty_strength
    # Embed logical biases
    for v, h in lin.items():
        q = logical_qubits[v]
        n = len(q)
        scal = 1.0 / n
        for qi in q:
            if qi not in avail_nodes:
                raise RuntimeError(f"Cannot place node {qi}")
            qac_lin[qi] = problem_scale * scal * h
    # Embed logical interactions
    for (u, v), j in qua.items():
        if not u < v:
            u, v = v, u
        uv_coupls = lq_inter_couplers[(u, v)]
        n = len(uv_coupls)
        scal = 1.0 / n
        for (qi, qj) in uv_coupls:
            e = (qi, qj) if qi < qj else (qj, qi)
            if e not in avail_edges:
                if strict:
                    print(f" ** Warning: cannot place physical edge ({qi}, {qj}) in logical edge ({u}, {v}) ")
                    return None, None
            else:
                qac_qua[e] = problem_scale * scal * j

    return qac_lin, qac_qua


def k4_nqac_nice2xy(t, x, z, u, a=1.0, x0=0.0, y0=0.0, lx=0.4, ly=0.35, ay=None):
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
    x = (float(xv) + s*lx)
    y = -(float(yv) + s*ly)
    if ay is None:
        ay = a
    return x0 + a*x, y0 + ay*y


def _decode_all_samples(sampleset: dimod.SampleSet, qac_map, bqm: BQM):
    vars = sampleset.variables
    n_samps = sampleset.record.sample.shape[0]
    n_logical = bqm.num_variables
    decoded_samples = np.zeros((n_samps, n_logical))
    decoding_ties = np.zeros((n_samps, n_logical))
    decoding_errs = np.zeros((n_samps, n_logical))

    for i, v in enumerate(bqm.variables):
        # logical qubit indices
        q = qac_map[v]
        idxs = np.asarray([vars.index(qi) for qi in q])
        q_values = sampleset.record.sample[:, idxs]  # [nsamples, physical_qubits_i]
        ql = np.sum(q_values, -1)
        decoded_samples[:, i] = np.where(ql > 0,  1, -1)  # ising decoding
        decoding_ties[:, i] = (ql == 0).astype(int)  # ties
        decoding_errs[:, i] = (np.abs(ql) != len(q)).astype(int)  # errors
    # break ties randomly
    ties = np.nonzero(decoding_ties)
    # ties = decoding_ties.astype(bool)
    if len(ties) > 0:
        r = np.random.randint(0, 2, len(ties[0]))*2 - 1
        decoded_samples[ties[0], ties[1]] = r

    energies = bqm.energies((decoded_samples, bqm.variables))

    return decoded_samples, energies, decoding_ties, decoding_errs


def _decode_all_k2_samples(sampleset: dimod.SampleSet, qac_map, bqm: BQM):
    vars = sampleset.variables
    n_samps = sampleset.record.sample.shape[0]
    n_logical = bqm.num_variables
    decoded_samples = np.zeros((2*n_samps, n_logical))
    decoding_ties = np.zeros((2*n_samps, n_logical))

    for i, v in enumerate(bqm.variables):
        # logical qubit indices
        q = qac_map[v]
        idxs = np.asarray([vars.index(qi) for qi in q])
        # first chain
        c1 = idxs[::2]
        # second chain
        c2 = idxs[1::2]
        q1 = sampleset.record.sample[:, c1]  # [nsamples, 2]
        q2 = sampleset.record.sample[:, c2]
        ql1 = np.sum(q1, -1)
        ql2 = np.sum(q2, -1)
        decoded_samples[:n_samps, i] = np.where(ql1 > 0,  1, -1)  # ising decoding
        decoded_samples[n_samps:, i] = np.where(ql2 > 0, 1, -1)
        decoding_ties[:n_samps, i] = (ql1 == 0).astype(int)  # ties
        decoding_ties[n_samps:, i] = (ql2 == 0).astype(int)
    # break ties randomly
    ties = np.nonzero(decoding_ties)
    # ties = decoding_ties.astype(bool)
    if len(ties) > 0:
        r = np.random.randint(0, 2, len(ties[0])) * 2 - 1
        decoded_samples[ties[0], ties[1]] = r

    energies = bqm.energies((decoded_samples, bqm.variables))

    return decoded_samples, energies, decoding_ties


class PegasusK4NQACGraph(AbstractQACGraph):
    def __init__(self, m, node_list, edge_list, strict=True, k2_mode=False, purge_deg1=True):
        """
         Level 1 Nested QAC graph with K4 embedding
         :param m:
         :param edge_list:
         :return:
         """
        super(PegasusK4NQACGraph, self).__init__()
        node_set = set(node_list)
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
        # for t in range(3):
        #     for x in range(m - 2):
        #         for z in range(m - 1):
        #             for u in range(2):
        #                     qac_edges.add(((t, x, z, u), (t, x + 1, z, u)))
        # Vertical-directed external bonds
        #  (t, x, z, u)  -- (t, x, z+1, u),  0 <= z < m-2 except (t=0,z=0)
        # for t in range(3):
        #     for x in range(m - 1):
        #         for z in range(m - 2):
        #             #if not (t == 0 and z == 0):
        #             for u in range(2):
        #                 qac_edges.add(((t, x, z, u), (t, x, z + 1, u)))

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
        logical_qubit_qpu_map = {}
        for k, v in logical_qubit_map.items():
            avail_qubits = [q for q in v if q in node_set]
            logical_qubit_qpu_map[k] = avail_qubits
        # Determine the physical couplers required to embed the logical graph
        # A subgraph of K4 with at least 3 nodes and 3 edges must be connected, so we accept with this criterion
        # If the graph is strict, then require all K4 nodes and edges to exist
        def accept_qubit_crit(q, lc):
            if k2_mode:
                return len(lc) >= 2
            if strict:
                return len(q) == 4 and len(lc) == 6
            else:
                return len(q) >= 3 and len(lc) >= 3
        def accept_coupler_crit(q1, q2, lc):
            if k2_mode:
                return len(lc) >= 2
            if strict:  # Logical couplings must consist of at least 4 physical couplings
                return len(lc) >= 4
            else:  # Otherwise, accept coupling with at least 2 physical couplings
                return len(lc) >= 2
        lq_intra_couplers, lq_inter_couplers = embed_logical_qubits(
            logical_qubit_map, qac_edges, edge_set, strict=False, direct_coupling=k2_mode,
            accept_qubit_crit=accept_qubit_crit,
            accept_coupler_crit=accept_coupler_crit
        )

        # self.node_embeddings = lq_intra_couplers
        # self.edge_embeddings = lq_inter_couplers
        logical_nodes = set(lq_intra_couplers.keys())
        logical_edges = set(lq_inter_couplers.keys())
        self.nodes = logical_nodes
        self.edges = logical_edges
        self.node_qubit_map = logical_qubit_qpu_map
        self.node_intra_couplers = lq_intra_couplers
        self.edge_inter_couplers = lq_inter_couplers
        self.qubit_array = qac_qubits
        # Make the networkx graph
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        for n in self.nodes:
            g.nodes[n]['embedding'] = lq_intra_couplers[n]
        for e in self.edges:
            g.edges[e]['embedding'] = lq_inter_couplers[e]
            g.edges[e]['weight'] = len(lq_inter_couplers[e])
        self.g = g
        if purge_deg1:
            self.purge_deg1()

    @classmethod
    def from_sampler(cls, m, sampler, **kwargs):
        nodes = set(sampler.nodelist)
        edges = set(sampler.edgelist)
        return cls(m, nodes, edges, **kwargs)

    def draw(self, draw_nodes=None, a=60.0, ay=None, lx=0.4, ly=0.35, return_graph=False, **draw_kwargs):
        if 'nodelist' in draw_kwargs:
            draw_nodes = draw_kwargs['nodelist']
        else:
            draw_nodes = self.nodes
        pos_list = {node: k4_nqac_nice2xy(*node, a=a, lx=lx, ly=ly, ay=ay) for node in draw_nodes}
        g = self.g
        nx.draw_networkx(g, pos=pos_list, with_labels=False, font_size=12, **draw_kwargs)
        if return_graph:
            g2 = nx.Graph()
            g2.add_nodes_from(list(sorted(pos_list.keys())))
            g_sub = g.subgraph(list(g2.nodes))
            g2.add_edges_from(g_sub.edges)

            for u in g2.nodes:
                g2.nodes[u]['position'] = pos_list[u]

            for u, v in g2.edges:
                if u in pos_list and v in pos_list:
                    g2.edges[u, v]['distance'] = np.abs(np.asarray(pos_list[u]) - np.asarray(pos_list[v]))
                else:
                    g2.edges[u, v]['distance'] = None
            return g2


class PegasusNQACEmbedding(AbstractQACEmbedding):
    def __init__(self, m, child_sampler, nqac_graph: PegasusK4NQACGraph):
        super(PegasusNQACEmbedding, self).__init__(m, child_sampler, nqac_graph)

    def validate_structure(self, bqm: BQM, penalty_strength=0.1):
        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
        lin, qua = try_embed_nqac_graph(bqm.linear, bqm.quadratic, self.qac_graph.node_qubit_map,
                                        self.qac_graph.node_intra_couplers,
                                        self.qac_graph.edge_inter_couplers,
                                        self._child_nodes, self._child_edges, penalty_strength)
        return lin, qua

    @bqm_structured
    def sample(self, bqm: BQM, qac_decoding=None, qac_penalty_strength=0.1, qac_problem_scale=1.0,
               qac_raw_samples=False, qac_clip=None, **parameters):
        """

        :param bqm:
        :param qac_decoding: 'qac' or 'c'
            Supported for K4 embedding:
                qac: Decode a K4 unit by majority vote, breaking ties randomly (Default)
                c: Extract two samples per readout by treating each K4 unit as two independent K2 chains

                The qac_graph must implement all intra- and inter- couplers consistent with either decoding mode
        :param qac_penalty_strength
        :param qac_problem_scale
        :param parameters:
        :return:
        """

        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
        lin, qua = try_embed_nqac_graph(bqm.linear, bqm.quadratic, self.qac_graph.node_qubit_map,
                                        self.qac_graph.node_intra_couplers,
                                        self.qac_graph.edge_inter_couplers,
                                        self._child_nodes, self._child_edges, qac_penalty_strength,
                                        problem_scale=qac_problem_scale)
        sub_bqm = AdjVectorBQM(lin, qua, bqm.offset, bqm.vartype)
        if qac_clip is not None:
            child_sampler = ClipComposite(self.child)
            parameters['lower_bound'] = - qac_clip
            parameters['upper_bound'] = qac_clip
        else:
            child_sampler = self.child
        # submit the problem
        sampleset: dimod.SampleSet = child_sampler.sample(sub_bqm, **parameters)
        if qac_decoding == 'c':
            decode_samp = self._extract_k2_solutions(sampleset, bqm, include_raw=qac_raw_samples)
        else:
            decode_samp = self._extract_qac_solutions(sampleset, bqm, include_raw=qac_raw_samples)
        return decode_samp

    def _extract_qac_solutions(self, sampleset: dimod.SampleSet, bqm: BQM, include_raw=False):
        samples, energies, q_ties, q_errs = _decode_all_samples(sampleset, self.qac_graph.node_qubit_map, bqm)

        num_occurrences = sampleset.data_vectors['num_occurrences']
        info = sampleset.info

        vectors = {}
        if include_raw:
            vectors['errors'] = q_errs
            vectors['ties'] = q_ties
        vectors["error_p"] = np.mean(q_errs, -1)
        vectors["tie_p"] = np.mean(q_ties, -1)
        sub_sampleset = dimod.SampleSet.from_samples((samples, bqm.variables), sampleset.vartype, energies,
                                                     info=info, num_occurrences=num_occurrences, **vectors)

        return sub_sampleset

    def _extract_k2_solutions(self, sampleset: dimod.SampleSet, bqm: BQM, include_raw=False):
        samples, energies, q_ties = _decode_all_k2_samples(sampleset, self.qac_graph.node_qubit_map, bqm)

        num_occurrences = sampleset.data_vectors['num_occurrences']
        # double up due to two-chain decoding
        num_occurrences = np.concatenate([num_occurrences, num_occurrences], axis=0)
        info = sampleset.info

        vectors = {}
        if include_raw:
            vectors['ties'] = q_ties
        vectors["tie_p"] = np.mean(q_ties, -1)
        sub_sampleset = dimod.SampleSet.from_samples((samples, bqm.variables), sampleset.vartype, energies,
                                                     info=info, num_occurrences=num_occurrences, **vectors)

        return sub_sampleset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import dwave_networkx as dnx
    from dwave.system import DWaveSampler

    l = 2
    k2_mode = True
    dws = DWaveSampler()

    qac_graph = PegasusK4NQACGraph(16, dws.nodelist, dws.edgelist, strict=False, k2_mode=k2_mode)
    fig, ax = plt.subplots(figsize=(12, 12))

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
    plt.savefig(f"pegasus_k4_nqac_logical_l{l}{'_k2' if k2_mode else ''}.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(30, 30))

    pegasus = dnx.pegasus_graph(16)
    # dnx.draw_pegasus(pegasus, ax=ax, with_labels=False, node_size=400,
    #                  node_color=[[0, 0, 0, 0]], edgecolors=[[0, 0, 0, 0]], crosses=True,
    #                  width=0.0, font_size=9)
    if k2_mode:
        for u, v, e in sub_qac.g.edges.data('embedding'):
            col = 'salmon'
            nds = qac_graph.node_qubit_map[u] + qac_graph.node_qubit_map[v]
            n=len(e)
            e_graph = dnx.pegasus_graph(16, node_list=nds, edge_list=e[:n//2])
            dnx.draw_pegasus(e_graph, ax=ax, node_color=[[0, 0, 0, 0]], edge_color=col, crosses=True, width=2)

        for u, d in sub_qac.g.nodes.data('embedding'):
            col = 'steelblue'
            nds = qac_graph.node_qubit_map[u]
            gq = dnx.pegasus_graph(16, node_list=nds[::2], edge_list=d)
            dnx.draw_pegasus(gq, node_size=400, ax=ax, node_color=col, edgecolors='black', edge_color=col, width=2, crosses=True)

            col = 'lightsteelblue'
            gq = dnx.pegasus_graph(16, node_list=nds[1::2], edge_list=d)
            dnx.draw_pegasus(gq, node_size=100, ax=ax, node_color=col, edgecolors=col, edge_color=col, width=2,
                             crosses=True)
    else:
        for u, v, e in sub_qac.g.edges.data('embedding'):
            col = 'salmon'
            nds = qac_graph.node_qubit_map[u] + qac_graph.node_qubit_map[v]
            e_graph = dnx.pegasus_graph(16, node_list=nds, edge_list=e)
            dnx.draw_pegasus(e_graph, ax=ax, node_color=[[0, 0, 0, 0]], edge_color=col, crosses=True, width=2)

        for u, d in sub_qac.g.nodes.data('embedding'):
            col = 'steelblue'
            nds = qac_graph.node_qubit_map[u]
            gq = dnx.pegasus_graph(16, node_list=nds, edge_list=d)
            dnx.draw_pegasus(gq, node_size=400, ax=ax, node_color=col, edgecolors='black', edge_color=col, width=2,
                             crosses=True)
    #
    # dnx.draw_pegasus(pegasus_3, ax=ax, with_labels=False, node_size=400,
    #                  node_color=[[0, 0, 0, 0]], edgecolors='black', crosses=True,
    #                  width=0.0, font_size=9, linewidths=2.0)

    plt.savefig(f"pegasus_k4_nqac_embed_l{l}{'_k2' if k2_mode else ''}.pdf")
