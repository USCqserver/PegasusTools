import networkx
import dwave_networkx


class ChimeraQubit:
    def __init__(self, l, x, y, z, k):
        """

        :param l: Chimera graph side length
        :param x, y: [0, L-1] 2D coordinates (column, row from upper left) of the cell
        :param z: 0 or 1, the lower or upper bipartition of the K_44 cell
        :param k: [0,3] the index within the bipartition
        """
        self.l = l
        self.x = x
        self.y = y
        self.z = z
        self.k = k

    def to_linear(self):
        return 8*(self.x + self.y * self.l) + 4*self.z + self.k

    def reduce(self, k2=0):
        q2 = ChimeraQubit(self.l, self.x, self.y, self.z, k2)
        return q2

    def opp(self):
        q2 = ChimeraQubit(self.l, self.x, self.y, 1-self.z, self.k)
        return q2


class ChimeraQACSpec:
    def __init__(self, q1, q2, q3, qpen):
        self.qphys = [q1, q2, q3]
        self.qpen = qpen

    def apply(self, f):
        ChimeraQACSpec(f(self.qphys[0]), f(self.qphys[1]), f(self.qphys[2]), f(self.qpen))


def qac_spec(q: ChimeraQubit):
    """
    Returns the QAC spec of the logical qubit whose physical qubits are in the same bipartition as q
    :param q:
    :return:
    """
    q2 = q.reduce()
    return ChimeraQACSpec(q2.reduce(0), q2.reduce(1), q2.reduce(2), q2.opp().reduce(3))


def generate_qac_chain_problem(l):

    mid_l = l // 2
    for x in range(l):
        y0 = mid_l if x%2 == 1 else 0