import typing
import numpy as np
import pandas as pd
from dimod import ComposedSampler, BinaryQuadraticModel, SampleSet


class EmbeddingSummaryWrapper(ComposedSampler):
    """
    Wrapper around  an embedding sampler to draw and save embeddings using
    the embedding context.
    """
    def __init__(self, embedding_sampler, embedding_name='embedding'):
        self._children = [embedding_sampler]
        self.embedding_name = embedding_name

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
        samples: SampleSet = self.child.sample(bqm, **parameters)
        emb_context = samples.info['embedding_context']

        emb = emb_context['embedding']
        variables = list(samples.variables)
        chain_lens = [len(emb[v]) for v in variables]
        emb_dat = {"variables": variables, "chain_length": chain_lens}
        emb_df = pd.DataFrame(emb_dat)
        info_df = samples.info.setdefault('dataframes', {})
        info_df[f"{self.embedding_name}_info"] = emb_df

        return samples