# distutils: language = c++
# cython: language_level = 3
from dimod import AdjVectorBQM, SPIN

def init_qac_penalty(qac_mapping, qac_qua: dict, penalty_strength):
    for _, q in qac_mapping:
        for qi in q[:3]:
            qac_qua[(qi, q[3])] = -penalty_strength


def embed_qac_graph(lin, qua, qac_dict: dict, penalty_strength: float, problem_scale:float =1.0):
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
    for v, q in qac_dict.items():
        for qi in q[:3]:
            qac_qua[(qi, q[3])] = -penalty_strength
    # Embed logical biases
    for v, h in lin.items():
        q = qac_dict[v]
        for qi in q[:3]:
            qac_lin[qi] = problem_scale * h
    # Embed logical interactions
    for (u, v), j in qua.items():
        qu = qac_dict[u]
        qv = qac_dict[v]
        for (qi, qj) in zip(qu[:3], qv[:3]):
            qac_qua[(qi, qj)] = problem_scale * j

    return qac_lin, qac_qua