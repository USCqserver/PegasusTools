# distutils: language = c++
# distutils: include_dirs = include
# cython: language_level = 3
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "pqubit.h" namespace "pgq":
    cdef cppclass qcell:
            int idxs[8]
    
    cdef cppclass Pqubit:
        int m
        int u, w, k, z
        Pqubit()
        int to_linear()
        Pqubit conn_external()
        Pqubit conn_odd()
        bool is_vert_coord()
        bool is_horz_coord()
        Pqubit conn_k44(int dk)
        
    cdef vector[qcell] generate_regular_cell_grid(int m)
