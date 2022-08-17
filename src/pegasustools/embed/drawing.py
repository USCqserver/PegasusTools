import typing
from itertools import product
import networkx as nx
import dwave_networkx as dnx
from dimod import ComposedSampler, BinaryQuadraticModel, SampleSet
from .structured import child_structure_dfs


def draw_minor_embedding(output_name, graph, results, bqm, embedding):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 12))

    # regenerate the embedded variables list
    varslist = []
    for v in results.variables:
        varslist += list(embedding[v])

    nodelist = []
    for v in results.variables:
        embv = embedding[v]
        for vi in embv:
            nodelist.append(vi)
    subgraph = nx.subgraph(graph, nodelist)

    edgelist = []
    for e in bqm.quadratic.keys():
        u, v = e
        embu = embedding[u]
        embv = embedding[v]
        coupls = []
        for vi, vj in product(embu, embv):
            if (vi, vj) in subgraph.edges:
                coupls.append((vi, vj))
        edgelist += coupls
        if len(coupls) == 0:
            raise ValueError

    # edgelist = list(nx.subgraph(qac_graph.g, nodelist).edges())
    dnx.draw_pegasus_embedding(subgraph, embedding, interaction_edges=bqm.quadratic.keys(),
                               unused_color=(0.2, 0.2, 0.2, 0.6), crosses=True,
                               node_size=16, width=0.4,  # alpha=0.8, width=0.8,
                               vmin=0, vmax=1)
    dnx.draw_pegasus_embedding(subgraph, embedding, interaction_edges=bqm.quadratic.keys(),
                               crosses=True, unused_color=None,
                               node_size=16, width=1.8, alpha=0.4,
                               vmin=0, vmax=1)
    plt.savefig(output_name)


class DrawEmbeddingWrapper(ComposedSampler):
    """
    Wrapper around  an embedding sampler to draw and save embeddings using
    the embedding context.
    """
    def __init__(self, embedding_sampler, drawing_output_name, embedding_name='embedding'):
        self._children = [embedding_sampler]
        self.call_count = 0
        self.drawing_output_name = drawing_output_name
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

    def sample(self, bqm: BinaryQuadraticModel,  **parameters) -> SampleSet:
        samples: SampleSet = self.child.sample(bqm, **parameters)
        structured_child = child_structure_dfs(self)
        child_graph = structured_child.to_networkx_graph()
        draw_minor_embedding(self.drawing_output_name + f'_{self.embedding_name}_{self.call_count}.pdf',
                             child_graph, samples, bqm,
                             samples.info['embedding_context']['embedding'])
        self.call_count += 1
        return samples
