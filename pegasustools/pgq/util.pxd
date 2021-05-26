# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.set cimport set
from libcpp.pair cimport pair

cdef class Adjacency:
    cdef set[int] nodes
    cdef set[pair[int,int]] edges

    