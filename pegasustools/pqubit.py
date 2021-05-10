"""
Descriptor class of a Pegasus qubit

Reference: Next-Generation Topology of D-Wave Quantum Processors, Boothby et. al. (2020)

Note: Qubits are referred to as vertical or horizontal by their addressing method,
not their illustrated topology. See Figs. 3,4 of Boothby et. al.

"""

import numpy as np
from typing import Union
from dimod import ComposedSampler, BinaryQuadraticModel, Sampler, StructureComposite, Structured, \
    bqm_structured
import dwave_networkx as dnx
import networkx as nx

Pegasus0Shift = [2, 2, 10, 10, 6, 6, 6, 6, 2, 2, 10, 10]


class Pqubit:
    def __init__(self, m, u, w, k, z, assert_valid=True):
        """
        Creates a new Pegasus qubit with the following coordinate values
        :param m: The size M of the Pegasus topology.
                  The complete graph has 24M(M-1) qubits, in which the largest connected subgraph has
                  8(3M-1)(M-1) main fabric qubits
        :param u: The qubit orientation, vertical (0) or horizontal (1)
        :param w: Perpendicular tile offset, i.e. the tile coordinate orthogonal to the direction of u
                  (the column index if u=0, or the row index if u=1)
                  0 <= w <= M-1
        :param k: Qubit offset
                  0 <= k <= 11
        :param z: Parallel tile offset, i.e. the index along the direction of u
                  (the row index if u=0, or the column index if u=1)
                  0 <= z <= M-2
        """
        check_str = self._check_if_not_valid(m, u, w, k, z)
        if assert_valid:
            if check_str is not None:
                raise ValueError(check_str)
            self._valid_coord = True
        else:
            if check_str is not None:
                self._valid_coord = False
                self._check_str = check_str
            else:
                self._valid_coord = True
                self._check_str = None

        self.m = m
        self.u = u
        self.w = w
        self.k = k
        self.z = z

    def __repr__(self):
        if not self._valid_coord:
            s = "!!!"
        else:
            s = ""
        if self.is_vert_coord():
            return s + f"Vert(M={self.m})[u=0, w: {self.w}, k: {self.k}, z: {self.z}]"
        else:
            return s + f"Horz(M={self.m})[u=1, w: {self.w}, k: {self.k}, z: {self.z}]"

    def __eq__(self, other):
        return (self.m == other.m and
                self.u == other.u and
                self.w == other.w and
                self.k == other.k and
                self.z == other.z
                )

    def none_if_invalid(self):
        if self._valid_coord:
            return self
        else:
            return None

    @staticmethod
    def _check_if_not_valid(m, u, w, k, z):
        if not m >= 1:
            return f"Invalid m: {m}. (Must be an integer greater than 0)"
        if not (u == 0 or u == 1):
            return f"Invalid u: {u}. (Valid range is 0 or 1)"
        if not (0 <= w <= m-1):
            return f"Invalid w: {w}. (Valid range is [0, {m - 1}] with m={m})"
        if not (0 <= k <= 11):
            return f"Invalid k: {k}. (Valid range is [0, 11])"
        if not (0 <= z <= m-2):
            return f"Invalid z: {z}. (Valid range is [0, {m - 2}] with m={m})"

        return None

    def to_linear(self):
        """
        Returns the linear index of this qubit in the graph
        :return:
        """
        if self._valid_coord:
            return self.z + (self.m - 1)*(self.k + 12*(self.w + self.m*self.u))
        else:
            raise RuntimeError(f"Attempted to convert from invalid coordinates:\n'{self._check_str}'")

    def conn_external(self, dz=1, **kwargs):
        """
        Returns the next qubit externally coupled to this one
        :return:
        """
        return Pqubit(self.m, self.u, self.w, self.k, self.z+dz, **kwargs)

    def conn_odd(self, **kwargs):
        """
        Returns the oddly coupled qubit to this one
        :return:
        """
        if self.k % 2 == 0:
            k2 = self.k + 1
        else:
            k2 = self.k - 1

        return Pqubit(self.m, self.u, self.w, k2, self.z, **kwargs)

    def is_vert_coord(self):
        return self.u == 0

    def is_horz_coord(self):
        return self.u == 1

    def conn_k44(self, dk, **kwargs):
        """
        Returns the qubit internally coupled to this one in the same K_44 subgraph at offset 0 <= dk <= 3
        :param dk:
        :return:
        """
        w2, k02, z2 = vert2horz(self.w, self.k, self.z) if self.is_vert_coord() else horz2vert(self.w, self.k, self.z)
        return Pqubit(self.m, 1-self.u, w2, k02 + dk, z2, **kwargs)

    def conn_internal(self, dk):
        """
        Returns the qubit internally coupled to this one at orthogonal offset dk, where the valid range of dk
        is  -6 <= dk <= 5
        This is equivalent to conn_k44 if dk is in the range [0, 3]. If dk is in [4, 5] then this connects to
        one orthogonally succeeding K_44 cluster. If dk is in [-4, -1], then this can connect to two orthogonally
        preceding clusters.
        If dk is in [-6, -5], then this connects to one second orthogonally preceeding cluster
        :param dk:
        :return:
        """
        _, k0_cluster, _ = vert2horz(self.w, self.k) if self.is_vert_coord() else horz2vert(self.w, self.k)
        j = (k0_cluster + dk) % 12
        w2, k2, z2 = internal_coupling(self.u, self.w, self.k, self.z, j)
        return Pqubit(self.m, 1 - self.u, w2, k2, z2)

    def conn_internal_abs(self, j, **kwargs):
        w2, k2, z2 = internal_coupling(self.u, self.w, self.k, self.z, j)
        return Pqubit(self.m, 1-self.u, w2, k2, z2, **kwargs)

    def k44_indices(self):
        """
        Creates a list of the 8 linear indices of the K44 cell containing this qubit
        By convention, a chimera K44 qubit index I%8 in [0, 8) is mapped to an index of a Pegasus K44 cell
        in ascending linear order, i.e.
          I%8:   0  1  2  3  4  5  6  7
            u:   0  0  0  0  1  1  1  1
          k%4:   0  1  2  3  0  1  2  3
        :return:
        """
        if self.is_vert_coord():
            dk_list = [0, 1, 2, 3]
            horz_q = [self.conn_k44(dk) for dk in dk_list]
            vert_q = [horz_q[0].conn_k44(dk) for dk in dk_list]
            qlist = vert_q + horz_q
            idx_list = [q.to_linear() for q in qlist]
            return idx_list
        else:
            dk_list = [0, 1, 2, 3]
            vert_q = [self.conn_k44(dk) for dk in dk_list]
            horz_q = [vert_q[0].conn_k44(dk) for dk in dk_list]
            qlist = vert_q + horz_q
            idx_list = [q.to_linear() for q in qlist]
            return idx_list


def vert2horz(w, k, z):
    """
    Gets the values of w, z, and k of the 4 vertical K_44 counterpart qubits
    to the vertical qubit in (u=0, w, k, z)
    """
    # Evaluate the raw XY coordinates from vertical coordinates
    t = k // 4
    xv = 3 * w + t
    yv = 2 + 3*z + (2 * t) % 3

    # Convert
    z2 = (xv - 1)//3
    w2 = yv // 3
    k02 = (yv % 3) * 4
    return w2, k02, z2


def horz2vert(w, k, z):
    """
    Gets values of w and z for the K_44 counterpart qubits
    to the horizontal qubit in (u=1, w, k, z)
    """
    #  Evaluate the raw XY coordinates from horizontal coordinates
    t = k // 4
    xh = 1 + 3*z + (2 * (t + 2)) % 3
    yh = 3 * w + t

    z2 = (yh - 2) // 3
    w2 = xh // 3
    k02 = (xh % 3) * 4
    return w2, k02, z2,


def internal_coupling(u, w, k, z, j):
    """
    Gets the internal coupling of opposite parity located at index j
    :param w:
    :param k:
    :param z:
    :return:
    """
    # d1 = 1 if j < Pegasus0Shift[k // 2] else 0
    # d2 = 1 if k < Pegasus0Shift[6 + (j // 2)] else 0
    # return z + d1, j, w - d2
    if u == 0:
        d1 = 1 if j < Pegasus0Shift[k // 2] else 0
        d2 = 1 if k < Pegasus0Shift[6 + (j // 2)] else 0
        return z+d1, j, w-d2
    else:
        d1 = 1 if k < Pegasus0Shift[(j // 2)] else 0
        d2 = 1 if j < Pegasus0Shift[6 + (k // 2)] else 0
        return z+d2, j, w-d1


def EmbedQACCoupling():
    pass


def collect_available_unit_cells(m, nodes_list):
    # nice coordinates of the graph
    # (t, y, x, u, k) where 0 <= t < 3, 0 <= x, y < M-1, u=0,1, 0<=k<=3
    w0 = [1, 0, 0]
    unit_cells = {}
    unavail_cells = 0
    # Iterate over unit cells in vertical coordinates
    for t in range(3):  # t = k // 4
        for w in range(w0[t], m - 1 + w0[t]):
            x = w - w0[t]
            for z in range(m - 1):
                k = 4 * t
                q0 = Pqubit(m, 0, w, k, z)
                k44_idxs = q0.k44_indices()
                graph_idxs = []
                for n in k44_idxs:
                    if n in nodes_list:
                        graph_idxs.append(n)
                    else:
                        # print(f"Skipping unit cell around (u=0,w={w},k={k},z={z})")
                        unavail_cells += 1
                        break
                if len(graph_idxs) == len(k44_idxs):
                    unit_cells[(t, x, z)] = graph_idxs

    return unit_cells, unavail_cells


class PegasusCellProblem(StructureComposite):

    children = None
    properties = None

    def __init__(self, m, child_sampler: Union[Structured], random_fill=None):
        logical_node_list = [i for i in range(7)]
        logical_edge_list = [(0, 4), (0, 5), (0, 6), (0, 7),
                             (1, 4), (1, 5), (1, 6), (1, 7),
                             (2, 4), (2, 5), (2, 6), (2, 7),
                             (3, 4), (3, 5), (3, 6), (3, 7)]

        system: nx.Graph = dnx.pegasus_graph(m, node_list=child_sampler.structure.nodelist,
                                   edge_list=child_sampler.structure.edgelist,
                                   nice_coordinates=True)

        unit_cells, unavail_cells = collect_available_unit_cells(m, system.nodes)

        print(f"Available cells in the topology: {len(unit_cells)}")
        print(f"Skipped cells: {unavail_cells}")
        self.unit_cells = unit_cells

        super(PegasusCellProblem, self).__init__(child_sampler, logical_node_list, logical_edge_list)

    @bqm_structured
    def sample(self, bqm: BinaryQuadraticModel, **parameters):
        """

        :param bqm:
        :param parameters:
        :return:
        """
        # [Decorator] Check that the problem can be embedded in an 8 qubit cell

        # Gather


def test_pqubit():
    # Assert that this makes a cycle
    q1 = Pqubit(3,  0, 0, 2, 0)  # u:0, w:0, k:2, z:0
    q2 = q1.conn_external()  # u:0, w:0, k:2,  z:1
    q3 = q2.conn_internal(-4)  # u:1, w:1, k:4, z:0
    q4 = q3.conn_internal(3)  # u:0 w:0, k:7, z:0
    q5 = q4.conn_odd()  # u:0, w:0, k:6, z:0
    q6 = q5.conn_internal(-2)  # u:1, w:1, k:2, z:0
    q7 = q6.conn_internal(0)  # u:0, w:0, k:8, z:0
    q8 = q7.conn_internal(-5)  # u:1, w:0, k:7, z:0
    q9 = q8.conn_internal(-2)  # u:0, w:0, k:2, z:0

    assert q1 == q9

    chim1 = q3.k44_indices()
    print()
    print(chim1)

    return


def test_pegasus_cell_problem():
    from dwave.system import DWaveSampler
    dws = DWaveSampler()
    problem = PegasusCellProblem(16, dws)

