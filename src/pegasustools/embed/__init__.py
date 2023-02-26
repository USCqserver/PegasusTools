import typing
import numpy as np
from dimod import ComposedSampler, BinaryQuadraticModel, SampleSet


def minor_embed_positions(tgt_positions, minor_embedding: dict):
    src_positions = {}
    for node, chain in minor_embedding.items():
        chain_positions = np.stack([np.asarray(tgt_positions[c]) for c in chain], axis=0)
        mean_pos = np.mean(chain_positions, axis=0)
        src_positions[node] = mean_pos
    return src_positions


class VariableMappingComposite(ComposedSampler):
    def __init__(self, child_sampler, variable_mapping=None):
        """

        :param child_sampler:
        :param variable_mapping: A tuple (n2l, l2n)
            where n2l (node to labels) is the inverse mapping and
            l2n (labels to nodes) is the forward mapping to the child nodes
        """
        self._children = [child_sampler]
        self.variable_mapping = variable_mapping

    @property
    def parameters(self) -> typing.Dict[str, typing.Any]:
        param = self.child.parameters.copy()
        return param

    @property
    def properties(self) -> typing.Dict[str, typing.Any]:
        return {'child_properties': self.child.properties.copy()}

    @property
    def children(self):
        return self._children

    def sample(self, bqm: BinaryQuadraticModel, **parameters) -> SampleSet:
        if self.variable_mapping is not None:
            n2l, l2n = self.variable_mapping
            mapping = {k: n for k, n in l2n.items() if k in bqm.linear}
            inverse_mapping = {n: k for (n, k) in n2l.items() if k in bqm.linear}
            bqmrel = bqm.relabel_variables(mapping, inplace=False)
        else:
            inverse_mapping = None

        samples: SampleSet = self.child.sample(bqmrel, **parameters)
        if inverse_mapping is not None:
            samples.relabel_variables(inverse_mapping, inplace=True)
        return samples
