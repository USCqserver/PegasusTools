import numpy as np
from typing import Union
from dimod.core.bqm import BQM
import dimod
from dimod import StructureComposite, Structured, bqm_structured, AdjVectorBQM
from .pqubit import collect_available_unit_cells, Pqubit
from .util.qac import init_qac_penalty



def try_embed_qac_graph(lin, qua, qac_dict: dict, avail_nodes, avail_edges, penalty_strength,
                        problem_scale=1.0, strict=True):
    """
    Linear and quadratic specifications should be Ising-type

    :param lin:
    :param qua:
    :param qac_dict: Maps variables to lists of four qubits
    :param penalty_strength:
    :param problem_scale:
    :return:
    """
    qac_lin = {}
    qac_qua = {}
    # Create the penalty Hamiltonian for all available QAC qubits
    for v in lin:
        q = qac_dict[v]
        for qi in q[:3]:
            qac_qua[(qi, q[3])] = -penalty_strength
    # Embed logical biases
    for v, h in lin.items():
        q = qac_dict[v]
        for qi in q[:3]:
            if qi not in avail_nodes:
                raise RuntimeError(f"Cannot place node {qi}")
            qac_lin[qi] = problem_scale * h
    # Embed logical interactions
    for (u, v), j in qua.items():
        qu = qac_dict[u]
        qv = qac_dict[v]
        for (qi, qj) in zip(qu[:3], qv[:3]):
            if (qi, qj) not in avail_edges:
                print(f" ** Warning: cannot place physical edge ({qi}, {qj}) in logical edge ({u}, {v}) ")
                if strict:
                    return None, None
            else:
                qac_qua[(qi, qj)] = problem_scale * j

    return qac_lin, qac_qua


def collect_pqac_graph(m, nodes_list, edge_list):
    """
    The PQAC graph is specified as follows:
        Within each cell with vertical and horizontal registers (qv, qh), the penalty qubits are placed in
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

    # Collect the penalty qubit indices
    # (t, x, z, u)
    qac_qubits = np.full((3, m-1, m-1, 2, 4), -1, dtype=np.int)
    for t in range(3):
        k = 4 * t
        for x in range(m-1):
            w = x + int(t == 0)
            for z in range(m-1):
                qp0 = Pqubit(m, 0, w, k, z)
                lp0 = np.asarray([qp0.conn_k44(1 + i).to_linear() for i in range(3)] + [qp0.to_linear()])
                qac_qubits[t, x, z, 0, :] = lp0
                if not (z == 0 and t == 0):  # Not valid for the top perimeter of cells
                    qp1 = qp0.conn_k44(0)
                    lp1 = np.asarray([qp1.conn_internal(5).to_linear(), qp1.conn_k44(2).to_linear(),
                                  qp1.conn_k44(3).to_linear(), qp1.to_linear()])
                    qac_qubits[t, x, z, 1, :] = lp1

    # Using the nice coordinates of the penalty qubit
    # Horizontal-directed external bonds
    #  (t, x, z, 0)  -- (t, x+1, z, 0)
    # Vertical-directed external bonds
    #  (t, x, z, 1)  -- (t, x, z+1, 1)
    # Diagonal internal bonds
    #  (t, x, z, 0)  -- (t+2, x, z, 1)
    #  (t, x, z, 0)  -- (t+1, x, z, 1)
    return qac_qubits


def _extract_qac_array(sampleset: dimod.SampleSet, qac_dict: dict, bqm: BQM):
    vars = sampleset.variables
    n_logical = bqm.num_variables
    arr_idxs = np.full([n_logical, 3], -1)
    for i, v in enumerate(bqm.variables):
        # logical qubit indices
        q = qac_dict[v]
        arr_idxs[i, :] = np.asarray([vars.index[qi] for qi in q[0:3]])

    q_values = sampleset.record.sample[:, arr_idxs]
    q_sum = np.sum(q_values, axis=2)
    # Majority vote reduction
    q_decode = np.where(q_sum > 0, 1, -1)

    return q_decode

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
                    print(f"Chain (t={t}, y={x}) placed at {ty_cells[0]}")

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
        problem_lin = {}
        problem_qua = {}
        used_cells = []
        # collect qac encodings
        for cell, qac_dict in self.avail_chains.items():
            bqm_lin = bqm.linear
            bqm_qua = bqm.quadratic
            lin, qua = try_embed_qac_graph(bqm_lin, bqm_qua, qac_dict, self._child_nodes, self._child_edges,
                                     qac_penalty_strength, qac_problem_scale)
            if lin is None:
                print(f" ** Cannot embed cell {cell}")
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
            split_results.append(_extract_qac_array(sampleset, qac_dict, bqm))
            # v_arr += [v] * nsamps
        samples_arr = np.concatenate(split_results, axis=0)
        # Evaluate the energies within each unit cell
        energy_arr = bqm.energies((samples_arr, bqm.variables))
        # v_arr = np.asarray(v_arr)
        sub_sampleset = dimod.SampleSet.from_samples(samples_arr, sampleset.vartype, energy_arr)

        return sub_sampleset

def test_pegasus_qac_problem():
    import pstats, cProfile
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

def test_pqac_gen():
    arr = collect_pqac_graph(3, None, None)
    return