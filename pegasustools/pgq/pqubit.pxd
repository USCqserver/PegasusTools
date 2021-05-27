# distutils: language = c++
# distutils: include_dirs = include
# cython: language_level = 3
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "pqubit.h" namespace "pgq" nogil:
    cdef cppclass qcell:
        int idxs[8]
        int& operator[](size_t i)
        const int& operator[](size_t i) const
    
    cdef cppclass Pqubit:
        int m
        int u, w, k, z
        Pqubit()
        Pqubit(int m, int u, int w, int k, int z)
        int to_linear()
        Pqubit conn_external()
        Pqubit conn_odd()
        bool is_vert_coord()
        bool is_horz_coord()
        Pqubit conn_k44(int dk)
        qcell k44_qubits()
        
    cdef cppclass CellGrid:
        int m
        vector[qcell] grid
        CellGrid()
        CellGrid(int m)
        const qcell& get(size_t t, size_t x, size_t z)

    cdef vector[qcell] generate_regular_cell_grid(int m)
