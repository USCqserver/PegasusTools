from typing import List, Dict
from dimod import BinaryQuadraticModel
#AdjacencyList = List[ Dict[int, float]]


def read_ising_adjacency(filename, max_k=1.0):
    """
    Reads a three-column text file specifying the adjacency
    :param filename:
    :return:
    """
    linear = {}
    quadratic = {}

    with open(filename) as f:
        for l, line in enumerate(f):
            toks = line.split()
            if len(toks) != 3:
                raise ValueError(f"Expected three tokens in line {l}")
            i, j, K = int(toks[0]), int(toks[1]), float(toks[2])
            if i == j:
                linear[i] = K / max_k
            else:
                (i2, j2) = (i, j) if i < j else (j, i)
                quadratic[(i2, j2)] = K / max_k

    bqm = BinaryQuadraticModel.from_ising(linear, quadratic)
    return bqm
