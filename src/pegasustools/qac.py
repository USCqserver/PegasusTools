import os
import pickle
import numpy as np
from typing import Union
from dimod import BQM
import networkx as nx
import dimod
from dimod import StructureComposite, Structured, bqm_structured, AdjVectorBQM, ComposedSampler
from pegasustools.embed.structured import PgtStructured
from pegasustools.pqubit import collect_available_unit_cells, Pqubit
from pegasustools.util.graph import random_walk_loop


def qac_nice2xy(t, x, z, u, a=1.0, x0=0.0, y0=0.0):
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
    x = (float(xv) + s*0.2)
    y = -(float(yv) - s*0.2)

    return x0 + a*x, y0 + a*y


def embed_qac_graph(lin, qua, qac_map, penalty_strength,
                        problem_scale=1.0, strict=True):
    """
    Embed the qac graph according to the mapping without checking available nodes and edges
    on the physical graph
    Linear and quadratic specifications should be Ising-type

    :param lin:
    :param qua:
    :param qac_map: Maps variables to lists of four qubits
    :param penalty_strength:
    :param problem_scale:
    :return:
    """
    qac_lin = {}
    qac_qua = {}
    # Create the penalty Hamiltonian for all available QAC qubits
    for v in lin:
        q = qac_map[v]
        for qi in q[:3]:
            qac_qua[(qi, q[3])] = -penalty_strength
    # Embed logical biases
    for v, h in lin.items():
        q = qac_map[v]
        for qi in q[:3]:
            qac_lin[qi] = problem_scale * h
    # Embed logical interactions
    for (u, v), j in qua.items():
        qu = qac_map[u]
        qv = qac_map[v]
        for (qi, qj) in zip(qu[:3], qv[:3]):
            qac_qua[(qi, qj)] = problem_scale * j

    return qac_lin, qac_qua


def purge_deg1(g: nx.Graph):
    """ Iteratively purge trivial degree 1 nodes"""
    deg1_nodes = [n for n in g.nodes if g.degree[n] < 2]
    n_purge = 0
    while len(deg1_nodes) > 0:
        g.remove_nodes_from(deg1_nodes)
        n_purge += len(deg1_nodes)
        deg1_nodes = [n for n in g.nodes if g.degree[n] < 2]
    return g, n_purge


def _assert_penalty_edges(q, avail_edges):
    qp = q[3]
    for qi in q[:3]:
        e = (qi, qp) if qi < qp else (qp, qi)
        if e not in avail_edges:
            return False
    return True


def try_embed_qac_graph(lin, qua, qac_map, avail_nodes, avail_edges, penalty_strength,
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
        q = qac_map[v]
        #_assert_penalty_edges(q, avail_edges)
        qp = q[3]
        for qi in q[:3]:
            e = (qi, qp) if qi < qp else (qp, qi)
            if e not in avail_edges:
                raise RuntimeError(f"Cannot place penalty coupling {e}")
            qac_qua[e] = -penalty_strength
    # Embed logical biases
    for v, h in lin.items():
        q = qac_map[v]
        for qi in q[:3]:
            if qi not in avail_nodes:
                raise RuntimeError(f"Cannot place node {qi}")
            qac_lin[qi] = problem_scale * h
    # Embed logical interactions
    for (u, v), j in qua.items():
        qu = qac_map[u]
        qv = qac_map[v]
        for (qi, qj) in zip(qu[:3], qv[:3]):
            e = (qi, qj) if qi < qj else (qj, qi)
            if e not in avail_edges:
                #print(f" ** Warning: cannot place physical edge ({qi}, {qj}) in logical edge ({u}, {v}) ")
                if strict:
                    print(f" ** Warning: cannot place physical edge ({qi}, {qj}) in logical edge ({u}, {v}) ")
                    return None, None
            else:
                qac_qua[e] = problem_scale * j

    return qac_lin, qac_qua


class AbstractQACGraph:
    def __init__(self):
        self.g = None
        self.qubit_array = None
        self.nodes = None
        self.edges = None
        self.node_intra_couplers = None
        self.edge_inter_couplers = None
        #self.node_embeddings = None
        #self.edge_embeddings = None

    @classmethod
    def from_sampler(cls, m, sampler):
        raise NotImplementedError

    @property
    def node_embeddings(self):
        raise NotImplementedError

    @property
    def edge_embeddings(self):
        raise NotImplementedError

    def purge_deg1(self):
        """ Iteratively purge trivial degree 1 nodes"""
        g, n_purge = purge_deg1(self.g)
        self.g = g
        self.nodes = set(self.g.nodes)
        self.edges = set(self.g.edges)
        print(f"Purged {n_purge} trivial nodes")

    def draw(self, **draw_kwargs):
        pos_list = {node: qac_nice2xy(*node, a=60.0) for node in self.nodes}
        g = self.g
        nx.draw_networkx(g, pos=pos_list, with_labels=False, font_size=12, **draw_kwargs)

    def subtopol(self, l, purge_deg1=True):
        sub_nodes = set([n for n in self.nodes if n[1] < l and n[2] < l])
        g2 = self.g.copy()
        g2.remove_nodes_from(n for n in self.g if n not in set(sub_nodes))
        # g2: nx.Graph = nx.subgraph(self.g, nbunch=sub_nodes).copy()
        sub_qubits = self.qubit_array[:, :l, :l]

        sub_qac = self.__new__(self.__class__)
        sub_qac.nodes = sub_nodes
        sub_qac.edges = set(g2.edges)
        sub_qac.g = g2
        sub_qac.qubit_array = sub_qubits
        if purge_deg1:
            sub_qac.purge_deg1()

        return sub_qac

    def save(self, cache_path):
        print(f"Saving qac graph to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, cache_path):
        print(f"Loading from {cache_path}")
        with open(cache_path, 'rb') as f:
            _qac_graph: cls = pickle.load(f)
        return _qac_graph

    @classmethod
    def load_or_initialize(cls, cache_path, *args, save=True, **kwargs):
        if os.path.isfile(cache_path):
            qac_graph = cls.load(cache_path)
        else:
            qac_graph = cls.__init__(*args, **kwargs)
            if save:
                print(f"Saving qac graph to {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(qac_graph, f)
        return qac_graph


class PegasusQACGraph(AbstractQACGraph):
    def __init__(self, m, nodes_list, edge_list, strict=True, purge_deg1=True):
        """
         The PQAC graph is specified as follows:
             Within each cell with vertical (u=0) and horizontal (u=1) registers (qv, qh), the penalty qubits are placed in
             qv[0] and qh[0]
             The corresponding logical qubits are
                 qv[0]  -- qh[1:4]
                 qh[1]  -- qh[1].conn_internal(5), qv[1], qv[2]
             The qubits in a complete logical graph all have a degree of 4 except at the boundary qubits
             The available logical couplings to each logical qubit are
                 Cell coupling
                 External coupling

         :param m:
         :param nodes_list:
         :param edge_list:
         :return:
         """
        super(PegasusQACGraph, self).__init__()
        # Collect the penalty qubit indices
        # (t, x, z, u)

        qac_nodes = set()
        qac_qubits = np.full((3, m - 1, m - 1, 2, 4), -1, dtype=int)
        for t in range(3):
            k = 4 * t
            for x in range(m - 1):
                w = x + int(t == 0)
                for z in range(m - 1):
                    qp0 = Pqubit(m, 0, w, k, z)
                    lp0 = np.asarray([qp0.conn_k44(1 + i).to_linear() for i in range(3)] + [qp0.to_linear()])
                    qac_qubits[t, x, z, 0, :] = lp0
                    if all(q in nodes_list for q in lp0) and _assert_penalty_edges(lp0, edge_list):
                        qac_nodes.add((t, x, z, 0))
                    if not (z == 0 and t == 0):  # Not valid for the top perimeter of cells
                        qp1 = qp0.conn_k44(0)
                        lp1 = np.asarray([qp1.conn_internal(5).to_linear(), qp1.conn_k44(2).to_linear(),
                                          qp1.conn_k44(3).to_linear(), qp1.to_linear()])
                        qac_qubits[t, x, z, 1, :] = lp1
                        if all(q in nodes_list for q in lp1) and _assert_penalty_edges(lp1, edge_list):
                            qac_nodes.add((t, x, z, 1))
        edges = set()

        def connect_if_avail(e1, e2):
            if e1 in qac_nodes and e2 in qac_nodes:
                q1 = qac_qubits[e1]
                q2 = qac_qubits[e2]
                nedges = 0
                for qi, qj in zip(q1[:3], q2[:3]):
                    e = (qi, qj) if qi < qj else (qj, qi)
                    if e in edge_list:
                        nedges += 1
                if strict and nedges == 3:
                    edges.add((e1, e2))
                elif (not strict) and nedges > 0:
                    edges.add((e1, e2))

        # Using the nice coordinates of the penalty qubit
        # Furthermore, all connected node pairs are ordered lexicographically
        # Horizontal-directed external bonds
        #  (t, x, z, 0)  -- (t, x+1, z, 0),  0 <= x < m-2
        for t in range(3):
            for x in range(m - 2):
                for z in range(m - 1):
                    connect_if_avail((t, x, z, 0), (t, x + 1, z, 0))
        # Vertical-directed external bonds
        #  (t, x, z, 1)  -- (t, x, z+1, 1),  0 <= z < m-2 except (t=0,z=0)
        for t in range(3):
            for x in range(m - 1):
                for z in range(m - 2):
                    if not (t == 0 and z == 0):
                        connect_if_avail((t, x, z, 1), (t, x, z + 1, 1))

        # Cluster bonds
        #  (t, x, z, 0)  -- (t, x, z, 1),
        for t in range(3):
            for x in range(m - 1):
                for z in range(m - 1):
                    connect_if_avail((t, x, z, 0), (t, x, z, 1))
        # Diagonal internal bonds
        #  (0, x, z, 0)  -- (2, x, z, 1)
        #  (2, x, z, 0)  -- (1, x, z, 1)
        #  (1, x, z, 0)  -- (0, x-1, z+1, 1)
        for x in range(m - 1):
            for z in range(m - 1):
                connect_if_avail((0, x, z, 0), (2, x, z, 1))
                connect_if_avail((1, x, z, 1), (2, x, z, 0))
        for x in range(1, m - 1):
            for z in range(m - 2):
                connect_if_avail((0, x - 1, z + 1, 1), (1, x, z, 0))
        # Cross-diagonal bonds
        #  (0, x, z, 0)  -- (1, x+1, z, 1)
        #  (2, x, z, 0)  -- (0, x, z+1, 1)
        #  (1, x, z, 0)  -- (2, x, z+1, 1)
        for x in range(m - 2):
            for z in range(m - 1):
                connect_if_avail((0, x, z, 0), (1, x + 1, z, 1))
        for x in range(m - 1):
            for z in range(m - 2):
                connect_if_avail((0, x, z + 1, 1), (2, x, z, 0))
                connect_if_avail((1, x, z, 0), (2, x, z + 1, 1))

        self.nodes = qac_nodes
        self.edges = edges
        self.qubit_array = qac_qubits
        # Make the networkx graph
        g = nx.Graph()
        g.add_nodes_from(qac_nodes)
        g.add_edges_from(edges)
        self.g = g
        if purge_deg1:
            self.purge_deg1()

    @classmethod
    def from_sampler(cls, m, sampler, **kwargs):
        nodes = set(sampler.nodelist)
        edges = set(sampler.edgelist)
        return cls(m, nodes, edges, **kwargs)

def _extract_all_samples(sampleset:  dimod.SampleSet, qac_map, bqm: BQM, ancilla=False):
    vars = sampleset.variables
    n_logical = bqm.num_variables
    arr_idxs = np.full([n_logical, 3], -1)
    p = 4 if ancilla else 3
    for i, v in enumerate(bqm.variables):
        # logical qubit indices
        q = qac_map[v]
        arr_idxs[i, :] = np.asarray([vars.index(qi) for qi in q[0:p]])

    q_values = sampleset.record.sample[:, arr_idxs]
    # [nsamples, nlogical, 3]
    return q_values


def _decode_qac_array(sampleset: dimod.SampleSet, qac_map, bqm: BQM):
    q_values = _extract_all_samples(sampleset, qac_map, bqm, ancilla=False)
    q_sum = np.sum(q_values, axis=2)
    # Majority vote reduction
    q_decode = np.where(q_sum > 0, 1, -1)
    q_err = np.where(np.abs(q_sum) == 3, 0.0, 1.0)

    energies = bqm.energies((q_decode, bqm.variables))

    return q_decode, energies, q_err


def _decode_all_array(sampleset: dimod.SampleSet, qac_map, bqm: BQM, ancilla=False):
    """
    Return all copies of the instance and their energies
    :param sampleset:
    :param qac_map:
    :param bqm:
    :param ancilla:
    :return:
    """
    p = 4 if ancilla else 3
    q_values = _extract_all_samples(sampleset, qac_map, bqm, ancilla=ancilla)
    q_values = np.transpose(q_values, [0, 2, 1])  # [nsamples, 3, nlogical]
    nsamps = q_values.shape[0]
    q_values = np.reshape(q_values, [nsamps*p, -1])
    energies = bqm.energies((q_values, bqm.variables))
    num_occurrences = sampleset.data_vectors['num_occurrences']
    num_occurrences_all = np.tile(num_occurrences, (p, 1)).transpose().flatten()
    return q_values, energies, num_occurrences_all


def _decode_c_array(sampleset: dimod.SampleSet, qac_map, bqm: BQM):
    """
    Decode according to the classical strategy (C), decoding the copy out of four with the smallest energy
    :param sampleset:
    :param qac_map:
    :param bqm:
    :return:
    """
    q_samps = _extract_all_samples(sampleset, qac_map, bqm, ancilla=True)
    nsamps = q_samps.shape[0]
    # Find the smallest energy of the three copies per sample
    q_values = np.transpose(q_samps, (1, 2))  # [nsamples, 3, nlogical]
    q_values = np.reshape(q_values, [nsamps * 3, -1])
    energies = bqm.energies((q_values, bqm.variables))
    energies = np.reshape(energies, [nsamps, 3])
    min_idxs = np.argmin(energies, axis=-1)

    min_q_samps = np.take_along_axis(q_samps, np.reshape(min_idxs, [nsamps, 1, 1]), axis=-1)
    min_energies = np.take_along_axis(energies, np.reshape(min_idxs, [nsamps, 1, 1]), axis=-1)
    return min_q_samps, min_energies


class AbstractQACEmbedding(StructureComposite, PgtStructured):
    def __init__(self, m, child_sampler, qac_graph: AbstractQACGraph):

        child_nodes = set(child_sampler.nodelist)
        child_edges = set(child_sampler.edgelist)

        logical_nodes = list(qac_graph.nodes)
        logical_edges = list(qac_graph.edges)
        super().__init__(child_sampler, logical_nodes, logical_edges)
        self.m = m
        self._child_nodes = child_nodes
        self._child_edges = child_edges
        self.qac_graph = qac_graph

    def networkx_graph(self) -> nx.Graph:
        return self.qac_graph.g


class PegasusQACEmbedding(AbstractQACEmbedding):
    def __init__(self, m, child_sampler, cache=True):
        cache_path = f".pegasus_qac_embedding.dat"
        child_nodes = set(child_sampler.nodelist)
        child_edges = set(child_sampler.edgelist)
        if cache and os.path.isfile(cache_path):
            qac_graph = PegasusQACGraph.load(cache_path)
        else:
            qac_graph = PegasusQACGraph(m, child_nodes, child_edges, strict=True)
        super(PegasusQACEmbedding, self).__init__(m, child_sampler, qac_graph)

    def validate_structure(self, bqm: BQM):
        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
        lin, qua = try_embed_qac_graph(bqm.linear, bqm.quadratic, self.qac_graph.qubit_array,
                                       self._child_nodes, self._child_edges,
                                       penalty_strength=0.1, problem_scale=1.0,
                                       strict=True)

    @bqm_structured
    def sample(self, bqm: BQM, qac_decoding='qac', qac_penalty_strength=0.1, qac_problem_scale=1.0, **parameters):
        """

        :param bqm:
        :param qac_decoding: 'qac', 'c', or 'all'
        :param qac_penalty_strength
        :param qac_problem_scale
        :param parameters:
        :return:
        """
        if qac_decoding not in ['qac', 'c', 'all']:
            raise RuntimeError(f"Invalid QAC decoding option '{qac_decoding}'")

        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
        lin, qua = try_embed_qac_graph(bqm.linear, bqm.quadratic, self.qac_graph.qubit_array,
                                       self._child_nodes, self._child_edges,
                                   penalty_strength=qac_penalty_strength, problem_scale=qac_problem_scale, strict=True)
        sub_bqm = AdjVectorBQM(lin, qua, bqm.offset, bqm.vartype)
        # submit the problem
        sampleset: dimod.SampleSet = self.child.sample(sub_bqm, **parameters)

        return self._extract_qac_solutions(qac_decoding, sampleset, bqm)

    def _extract_qac_solutions(self, decoding, sampleset: dimod.SampleSet, bqm: BQM):
        q_err = None
        num_occurrences = sampleset.data_vectors['num_occurrences']
        info = sampleset.info
        if decoding == 'qac':
            samples, energies, q_err = _decode_qac_array(sampleset, self.qac_graph.qubit_array, bqm)
        elif decoding == 'c':
            samples, energies = _decode_c_array(sampleset, self.qac_graph.qubit_array, bqm)
        elif decoding == 'all':
            samples, energies, num_occurrences = _decode_all_array(sampleset, self.qac_graph.qubit_array, bqm, ancilla=False)
        else:
            raise RuntimeError(f" ** Invalid decoding option {decoding}")

        vectors = {}
        if q_err is not None:
            vectors['errors'] = q_err
            vectors["error_p"] = np.mean(q_err, axis=1)
        sub_sampleset = dimod.SampleSet.from_samples((samples, bqm.variables), sampleset.vartype, energies,
                                                     info=info, num_occurrences=num_occurrences, **vectors)

        return sub_sampleset


class PegasusQACChainEmbedding(StructureComposite):
    def __init__(self, m, child_sampler: Union[Structured]):
        """

        :param m:
        :param child_sampler:
        :param random_fill:
        """

        logical_node_list = [i for i in range(8)]
        logical_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4),
                             (4, 5), (5, 6), (6, 7)]
        self._child_nodes = set(child_sampler.nodelist)
        self._child_edges = set(child_sampler.edgelist)
        unit_cells, unavail_cells = collect_available_unit_cells(m, self._child_nodes, self._child_edges,
                                                                 check='qac', register=0)
        avail_chains = {}
        # Find contiguous chains of 8 logical qubits in the x direction, going through t and y coordinates
        for t in range(3):
            for x in range(m-1):
                ty_cells = []
                for z in range(m-1):
                    #alternate available chains as much as possible
                    if (t + x) % 2 == 1:
                        z2 = m-1 - z
                    else:
                        z2 = z
                    v = (t, x, z2)
                    if v in unit_cells:
                        ty_cells.append(v)
                        if len(ty_cells) == 8:
                            break
                    else:
                        ty_cells.clear()
                        continue
                if (t + x) % 2 == 1:
                    ty_cells.reverse()
                if len(ty_cells) == 8:
                    #print(f"Chain (t={t}, y={x}) placed at {ty_cells[0]}")
                    ty_dict = {i: unit_cells[q] for i,q in enumerate(ty_cells)}
                    avail_chains[ty_cells[0]] = ty_dict

        self.avail_chains = avail_chains
        super().__init__(child_sampler, logical_node_list, logical_edge_list)

    @bqm_structured
    def sample(self, bqm: BQM, qac_penalty_strength=0.1, qac_problem_scale=1.0, **parameters):
        """
        :param bqm:
        :return:
        """
        bqm_lin = bqm.linear
        bqm_qua = bqm.quadratic
        problem_lin = {}
        problem_qua = {}
        used_cells = []
        # collect qac encodings
        for cell, qac_dict in self.avail_chains.items():
            lin, qua = try_embed_qac_graph(bqm_lin, bqm_qua, qac_dict, self._child_nodes, self._child_edges,
                                     qac_penalty_strength, qac_problem_scale)
            if lin is None:
                pass
                #print(f" ** Cannot embed cell {cell}")
            else:
                problem_lin.update(lin)
                problem_qua.update(qua)
                used_cells.append(cell)

        sub_bqm = AdjVectorBQM(problem_lin, problem_qua, bqm.offset, bqm.vartype)
        # submit the problem
        sampleset: dimod.SampleSet = self.child.sample(sub_bqm, **parameters)
        sub_sampleset = self._extract_qac_solutions(used_cells, sampleset, bqm)
        return sub_sampleset

    def _extract_qac_solutions(self, used_cells, sampleset: dimod.SampleSet, bqm: BQM):
        split_results = []
        for cell in used_cells:
            qac_dict = self.avail_chains[cell]
            split_results.append(_decode_qac_array(sampleset, qac_dict, bqm))
            # v_arr += [v] * nsamps
        samples_arr = np.concatenate(split_results, axis=0)
        # Evaluate the energies within each unit cell
        energy_arr = bqm.energies((samples_arr, bqm.variables))
        # v_arr = np.asarray(v_arr)
        sub_sampleset = dimod.SampleSet.from_samples(samples_arr, sampleset.vartype, energy_arr)

        return sub_sampleset


def test_pegasus_qac_problem():
    import pstats
    import cProfile
    from dwave.system import DWaveSampler
    dws = DWaveSampler()
    print("Profiling cell embedding...")

    cProfile.runctx("PegasusQACChainEmbedding(16, dws)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumulative").print_stats()
    bqm = dimod.AdjVectorBQM.from_ising({3: 0.4,}, {(0, 1): -1.0,
                                                    (1, 2): -1.0,
                                                    (2, 3): -1.0,
                                                    (3, 4): -1.0,
                                                    (4, 5): -1.0,
                                                    (5, 6): -1.0,
                                                    (6, 7): -1.0})
    sampler = PegasusQACChainEmbedding(16, dws)
    solution : dimod.SampleSet = sampler.sample(bqm, qac_penalty_strength=0.1, qac_problem_scale=1.0 ,
                              answer_mode='raw', num_reads=64)
    solution = solution.aggregate()
    print(solution)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dwave.system import DWaveSampler

    dws = DWaveSampler()
    qac_graph = PegasusQACGraph(16, dws.nodelist, dws.edgelist)
    # qac, nodes, edges = collect_pqac_graph(16, dws.nodelist, dws.edgelist)
    #fig, ax = plt.subplots(figsize=(36, 36))
    fig, ax = plt.subplots(figsize=(8, 8))
    l = 8
    sub_qac = qac_graph.subtopol(l)
    n = np.random.randint(0, sub_qac.g.number_of_nodes())
    init_node = list(qac_graph.g.nodes)[n]
    rand_loop = random_walk_loop(init_node, qac_graph.g)
    print(rand_loop)
    sub_qac.draw()
    # Evaluate some statistics

    #plt.show()
    plt.savefig(f"pegasus_qac_logical_l{l}.png")
