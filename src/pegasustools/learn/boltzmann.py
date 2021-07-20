import random
import dwave.inspector
import dimod
import numpy as np
from typing import Union
from dimod import Structured, Sampler, AdjVectorBQM
from minorminer import find_embedding
from dwave.system import FixedEmbeddingComposite


class RestrictedBoltzmannMachineSampler(FixedEmbeddingComposite):
    def __init__(self, nv, nh, child_sampler: Union[Sampler, Structured]):
        self.nv = nv
        self.nh = nh
        logical_node_list = [i for i in range(nv + nh)]
        logical_edge_list = [(i, nv + j) for i in range(nv) for j in range(nh)]
        self._child_nodes = set(child_sampler.nodelist)
        self._child_edges = set(child_sampler.edgelist)

        embedding = find_embedding(logical_edge_list, self._child_edges)

        super().__init__(child_sampler, embedding=embedding)

    def sample_rbm(self, v: np.ndarray, h: np.ndarray, w: np.ndarray, **kwargs):
        """

        :param v: nv 1D array
        :param h: nh 1D array
        :param w: (nv x nv) matrix
        :param kwargs:
        :return:
        """
        lin = np.concatenate([v, h])
        zv = np.zeros((self.nv, self.nv))
        zh = np.zeros((self.nh, self.nh))
        qua = 0.5 * np.block([
            [zv,            w],
            [w.transpose(), zh]
        ])
        bqm = AdjVectorBQM(lin, qua, dimod.SPIN)

        return self.sample(bqm, **kwargs)


class ConvRBMSampler(FixedEmbeddingComposite):
    def __init__(self, nvx, nvy, nhx, nhy, kx, ky, nvc, nhc,
                 child_sampler: Union[Sampler, Structured]):
        self.nv = (nvx, nvy, nvc)
        self.nh = (nhx, nhy, nhc)
        self.k = (kx, ky, nvc, nhc)
        logical_node_list = [i for i in range(nv + nh)]
        logical_edge_list = [(i, nv + j) for i in range(nv) for j in range(nh)]
        self._child_nodes = set(child_sampler.nodelist)
        self._child_edges = set(child_sampler.edgelist)

        embedding = find_embedding(logical_edge_list, self._child_edges)

        super().__init__(child_sampler, embedding=embedding)


def test_rbm():
    from dwave.system import DWaveSampler

    nv = 32
    nh = 32
    dws = DWaveSampler()
    v = np.random.uniform(-0.2, 0.2, (nv,))
    h = np.random.uniform(-0.2, 0.2, (nh,))
    w = np.random.uniform(-0.5, 0.5, (nv, nh))

    rbm_sampler = RestrictedBoltzmannMachineSampler(nv, nh, dws)
    results = rbm_sampler.sample_rbm(v, h, w, num_reads=32)
    print(results.truncate(8))
    dwave.inspector.show(results)

