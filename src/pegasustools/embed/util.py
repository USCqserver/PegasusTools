import logging
import pathlib
import typing
import numpy as np
import pandas as pd
import pickle
from dimod import ComposedSampler, BinaryQuadraticModel, SampleSet
from dwave.system import LazyFixedEmbeddingComposite, FixedEmbeddingComposite

class EmbeddingSummaryWrapper(ComposedSampler):
    """
    Wrapper around  an embedding sampler to draw and save embeddings using
    the embedding context.
    """
    def __init__(self, embedding_sampler, embedding_name='embedding',
                 save_embedding=None):
        self._children = [embedding_sampler]
        self._collected_embeddings = []
        # check if the child sampler is a fixed embedding sampler
        if isinstance(embedding_sampler, LazyFixedEmbeddingComposite):
            self._child_is_fixed = True
        else:
            self._child_is_fixed = False
        self.embedding_name = embedding_name
        self.save_embedding = save_embedding

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

        emb = dict(emb_context['embedding'])
        self._collected_embeddings.append(emb)
        variables = list(samples.variables)
        chain_lens = [len(emb[v]) for v in variables]
        emb_dat = {"variables": variables, "chain_length": chain_lens}
        emb_df = pd.DataFrame(emb_dat)
        info_df = samples.info.setdefault('dataframes', {})
        info_df[f"{self.embedding_name}_info"] = emb_df
        if self.save_embedding is not None:
            nembds = len(self._collected_embeddings) # current number of embeddings
            if self._child_is_fixed:  # only need to save one embedding once
                if nembds == 1:
                    with open(self.save_embedding, 'wb') as f:
                        pickle.dump(emb, f)
                logging.info(f"Saved fixed embedding to {self.save_embedding}")
                self.save_embedding = None
            else:  # save a list of embeddings
                logging.info(f"Appending embedding {nembds - 1} to {self.save_embedding}")
                with open(self.save_embedding, 'wb') as f:
                    pickle.dump(self._collected_embeddings, f)
        return samples