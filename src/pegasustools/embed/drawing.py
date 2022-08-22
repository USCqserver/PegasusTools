import typing
import numpy as np
from itertools import product
import networkx as nx
import dwave_networkx as dnx
from dimod import BQM, ComposedSampler, BinaryQuadraticModel, SampleSet
from dimod.variables import Variables
from itertools import combinations, product
from pegasustools.nqac import PegasusNQACEmbedding, PegasusK4NQACGraph, AbstractQACEmbedding, AbstractQACGraph
from dwave_networkx.drawing.distinguishable_colors import distinguishable_color_map
from dwave_networkx.drawing import pegasus_layout
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



def draw_qac(output_name, qac_graph: AbstractQACGraph, results: SampleSet, bqm: BQM,
             embedding, color_errors=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 12))

    # regenerate the embedded variables list
    varslist = []
    for v in results.variables:
        varslist += list(embedding[v])
    emb_variables = Variables(varslist)
    nodecols = []
    nodelist = []
    if color_errors:
        mean_err_p = np.mean(results.record['errors'], axis=0)
    else:
        mean_err_p = np.zeros(len(varslist))

    for v in results.variables:
        embv = embedding[v]
        for vi in embv:
            nodelist.append(vi)
            nodecols.append(mean_err_p[emb_variables.index(vi)])
    edgelist = []
    for e in bqm.quadratic.keys():
        u, v = e
        embu = embedding[u]
        embv = embedding[v]
        coupls = []
        for vi, vj in product(embu, embv):
            if (vi, vj) in qac_graph.g.edges:
                coupls.append((vi, vj))
        edgelist += coupls
        if len(coupls) == 0:
            raise ValueError

    # edgelist = list(nx.subgraph(qac_graph.g, nodelist).edges())
    qac_graph.draw(node_size=25, alpha=0.8, width=0.8, nodelist=nodelist, edgelist=edgelist,
                   node_color=nodecols, cmap=plt.cm.get_cmap('bwr'), vmin=0, vmax=1)
    edgelist = []
    edgecols = []
    n = len(results.variables)
    cmap = distinguishable_color_map(int(n + 1))
    for i, v in enumerate(results.variables):
        embv = embedding[v]
        chain = []
        col = cmap(i/n)
        for (vi, vj) in combinations(embv, 2):
            if vi > vj:
                vi, vj = vj, vi
            e = (vi, vj)
            if e in qac_graph.g.edges:
                chain.append((vi, vj))
                edgecols.append(col)
        if len(embv) > 1 and len(chain) == 0:
            raise ValueError
        edgelist += chain
    qac_graph.draw(node_size=0.0, alpha=0.5, width=2.0, nodelist=nodelist, edgelist=edgelist,
                   node_color=[[1.0, 1.0, 1.0, 0.0]], edgecolors=[[1.0, 1.0, 1.0, 0.0]], linewidths=0.0,
                   edge_color=edgecols)
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
        embedding = samples.info['embedding_context']['embedding']
        structured_child = child_structure_dfs(self)
        child_graph = structured_child.to_networkx_graph()
        if child_graph.graph.get("family") == "pegasus":
            draw_minor_embedding(self.drawing_output_name + f'_{self.embedding_name}_{self.call_count}.pdf',
                                 child_graph, samples, bqm,
                                 embedding)
        elif isinstance(structured_child, AbstractQACEmbedding):
            draw_qac(self.drawing_output_name + f'_{self.embedding_name}_{self.call_count}.pdf',
                     structured_child.qac_graph, samples, bqm, embedding)
        self.call_count += 1
        return samples
