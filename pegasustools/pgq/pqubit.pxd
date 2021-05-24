# distutils: language = c++
# distutils: include_dirs = pegasustools/include
# cython: language_level = 3
from libcpp cimport bool

cdef extern from "pqubit.h" namespace "pgq":
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
