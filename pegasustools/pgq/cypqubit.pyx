# distutils: language = c++
# cython: language_level = 3
from cpython cimport array
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from pegasustools.pgq.pqubit cimport Pqubit, qcell, Pcoord
import array
from pegasustools.pgq.util import vert2horz, horz2vert, internal_coupling
from pegasustools.pgq cimport util

cdef bool in_nodes(int* q0, size_t nq, set[int] nodes):
    for i in range(nq):
        if nodes.find(q0[i]) == nodes.end():
            return False
    return True


cdef unordered_map[Pcoord, qcell] collect_available_unit_cells(int m, util.Adjacency adj):
    # nice coordinates of the graph
    # (t, y, x, u, k) where 0 <= t < 3, 0 <= x, y < M-1, u=0,1, 0<=k<=3
    cdef int w0[3] 
    cdef int x, k
    cdef Pqubit q0
    cdef qcell q
    cdef unordered_map[Pcoord, qcell] unit_cells
    cdef int unavail_cells = 0

    w0[:] = [1, 0, 0]
    
    # Iterate over unit cells in vertical coordinates
    for t in range(3):  # t = k // 4
        for w in range(w0[t], m - 1 + w0[t]):
            x = w - w0[t]
            for z in range(m - 1):
                k = 4 * t
                q0 = Pqubit(m, 0, w, k, z)
                q = q0.k44_qubits()
                if in_nodes(q.idxs, 8, adj.nodes):
                    unit_cells[Pcoord(t, x, z)] = q
                else:
                    unavail_cells += 1
                

    return unit_cells

cdef array.array a4 = array.array('i', [0, 1, 2, 3])
cdef int[:] i4 = a4

cdef struct cy_pqubit:
    int m
    int u, w, k, z


cpdef int to_linear(cy_pqubit q):
    """
            Returns the linear index of this qubit in the graph
            :return:
            """
    return q.z + (q.m - 1) * (q.k + 12 * (q.w + q.m * q.u))


cpdef _check_if_not_valid(cy_pqubit q):
    if not q.m >= 1:
        return f"Invalid m: {q.m}. (Must be an integer greater than 0)"
    if not (q.u == 0 or q.u == 1):
        return f"Invalid u: {q.u}. (Valid range is 0 or 1)"
    if not (0 <= q.w <= q.m - 1):
        return f"Invalid w: {q.w}. (Valid range is [0, {q.m - 1}] with m={q.m})"
    if not (0 <= q.k <= 11):
        return f"Invalid k: {q.k}. (Valid range is [0, 11])"
    if not (0 <= q.z <= q.m - 2):
        return f"Invalid z: {q.z}. (Valid range is [0, {q.m - 2}] with m={q.m})"

    return None


cdef class cyPqubit:
    cdef readonly int m
    cdef public int u, w, k, z
    cdef readonly int _valid_coord
    cdef readonly str _check_str

    def check_valid(self, raise_err=False):
        cdef cy_pqubit q2 = cy_pqubit(self.m, self.u, self.w, self.k, self.z)
        check_str = _check_if_not_valid(q2)
        if raise_err:
            if check_str is not None:
                raise ValueError(check_str)
            return None
        else:
            return check_str

    def __cinit__(self, int m, int u, int w, int k, int z):
        """

        """

        self.m = m
        self.u = u
        self.w = w
        self.k = k
        self.z = z


    def __init__(self, int m, int u, int w, int k, int z):
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
        pass

    def __repr__(self):
        val = self.check_valid(raise_err=False)
        if val is not None:
            s = "!!!"
        else:
            s = ""
        if self.is_vert_coord():
            return s + f"Vert(M={self.m})[u=0, w: {self.w}, k: {self.k}, z: {self.z}]"
        else:
            return s + f"Horz(M={self.m})[u=1, w: {self.w}, k: {self.k}, z: {self.z}]"

    def __eq__(self, cyPqubit other):
        return (self.m == other.m and
                self.u == other.u and
                self.w == other.w and
                self.k == other.k and
                self.z == other.z
                )

    def none_if_invalid(self):
        val = self.check_valid(raise_err=False)
        if val is None:
            return self
        else:
            return None

    cpdef to_linear(self):
        """
        Returns the linear index of this qubit in the graph
        :return:
        """
        return self.z + (self.m - 1)*(self.k + 12*(self.w + self.m*self.u))

    cpdef conn_external(self, int dz=1):
        """
        Returns the next qubit externally coupled to this one
        :return:
        """
        cdef cyPqubit q2 = cyPqubit(self.m, self.u, self.w, self.k, self.z + dz)
        return q2

    cpdef conn_odd(self):
        """
        Returns the oddly coupled qubit to this one
        :return:
        """
        if self.k % 2 == 0:
            k2 = self.k + 1
        else:
            k2 = self.k - 1
        cdef cyPqubit q2 = cyPqubit(self.m, self.u, self.w, k2, self.z)
        return q2

    cpdef is_vert_coord(self):
        return self.u == 0

    cpdef is_horz_coord(self):
        return self.u == 1

    cpdef conn_k44(self, int dk):
        """
        Returns the qubit internally coupled to this one in the same K_44 subgraph at offset 0 <= dk <= 3
        :param dk:
        :return:
        """
        w2, k02, z2 = vert2horz(self.w, self.k, self.z) if self.is_vert_coord() else horz2vert(self.w, self.k, self.z)
        cdef cyPqubit q2 = cyPqubit(self.m, 1 - self.u, w2, k02 + dk, z2)
        return q2

    cpdef conn_internal(self, dk):
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
        cdef cyPqubit q2 = cyPqubit(self.m, 1 - self.u, w2, k2, z2)
        return q2

    cpdef conn_internal_abs(self, j):
        w2, k2, z2 = internal_coupling(self.u, self.w, self.k, self.z, j)
        cdef cyPqubit q2 = cyPqubit(self.m, 1 - self.u, w2, k2, z2)
        return cyPqubit(self.m, 1 - self.u, w2, k2, z2)

    cpdef k44_indices(self):
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
            horz_q = [self.conn_k44(dk) for dk in a4]
            vert_q = [horz_q[0].conn_k44(dk) for dk in a4]
            qlist = vert_q + horz_q
            idx_list = [q.to_linear() for q in qlist]
            return idx_list
        else:
            vert_q = [self.conn_k44(dk) for dk in a4]
            horz_q = [vert_q[0].conn_k44(dk) for dk in a4]
            qlist = vert_q + horz_q
            idx_list = [q.to_linear() for q in qlist]
            return idx_list
