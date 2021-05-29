import numpy as np
from typing import Union
from dimod.core.bqm import BQM
from dimod import StructureComposite, Structured, bqm_structured, AdjArrayBQM
from .pqubit import collect_available_unit_cells, Pqubit
from .util.qac import embed_qac_graph, init_qac_penalty


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


    # Vertical-directed external bonds
    #  (t, x, z, 0)  -- (t, x, z+1, 0)
    # Horizontal-directed external bonds
    #  (t, x, z, 1)  -- (t, x+1, z, 1)
    
    return qac_qubits


class PegasusQACChainEmbedding(StructureComposite):
    def __init__(self, m, child_sampler: Union[Structured], penalty=0.1, random_fill=None):
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
                                                                 check='qac', register=1)
        avail_chains = {}
        # Find contiguous chains of 8 logical qubits in the x direction, going through t and y coordinates
        for t in range(3):
            for y in range(m-1):
                ty_cells = []
                for x in range(m-1):
                    #alternate available chains as much as possible
                    if (t + y) % 2 == 1:
                        x2 = m-1 - x
                    else:
                        x2 = x
                    v = (t, y, x2)
                    if v in unit_cells:
                        ty_cells.append(v)
                        if len(ty_cells) == 8:
                            break
                    else:
                        ty_cells.clear()
                        continue
                if (t + y) % 2 == 1:
                    ty_cells.reverse()
                if len(ty_cells) == 8:
                    print(f"Chain (t={t}, y={y}) placed at {ty_cells[0]}")
                    avail_chains[ty_cells[0]] = ty_cells

        super().__init__(child_sampler, logical_node_list, logical_edge_list)

    @bqm_structured
    def sample(self, bqm: BQM, **sample_kwargs):
        pass


def test_pegasus_qac_problem():
    import pstats, cProfile
    from dwave.system import DWaveSampler
    dws = DWaveSampler()
    print("Profiling cell embedding...")

    cProfile.runctx("PegasusQACChainEmbedding(16, dws)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumulative").print_stats()


def test_pqac_gen():
    arr = collect_pqac_graph(3, None, None)
    return