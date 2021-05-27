# distutils: language = c++
# cython: language_level = 3
from cpython cimport array
from pegasustools.pgq.pqubit cimport Pqubit, CellGrid, qcell, generate_regular_cell_grid
import array

__all__ = ['collect_complete_unit_cells']

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


def collect_complete_unit_cells(int grid_m, nodes_list, edge_list):
    cdef int m = grid_m
    #cdef CellGrid cell_grid = CellGrid(m)
    cdef Pqubit pq
    cdef qcell q
    cdef int unavail_cells = 0
    cdef int w0[3]
    cdef int t, w, k, x, z
    cdef int i, j
    unit_cells = {}
    w0[:] = [1, 0, 0]
    
    for t in range(3):
        k = 4*t
        for x in range(m-1):
            w = x + w0[t]
            for z in range(m-1):
                pq = Pqubit(m, 0, w, k, z)
                q = pq.k44_qubits()
                nodes = [q.idxs[i] for i in range(8)]
                edges = [(q.idxs[i], q.idxs[4 + j]) for i in range(4) for j in range(4)]
                if all(n in nodes_list for n in nodes):
                    if all(e in edge_list for e in edges):
                        unit_cells[(t, x, z)] = nodes
                        continue
                unavail_cells += 1

    return unit_cells, unavail_cells
