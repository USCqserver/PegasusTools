import argparse
import numpy as np
import networkx as nx
import pandas as pd
import yaml
import dimod
from dwave_networkx.drawing.distinguishable_colors import distinguishable_color_map
from itertools import combinations, product
from dimod.variables import Variables
from numpy.lib import recfunctions as rfn

from pegasustools.app import add_general_arguments, add_qac_arguments, run_sampler, save_cell_results
from pegasustools.qac import PegasusQACEmbedding
from pegasustools.nqac import PegasusNQACEmbedding, PegasusK4NQACGraph
from pegasustools.util.adj import read_ising_adjacency, read_mapping
from pegasustools.util.sched import interpret_schedule
from dwave.preprocessing import ScaleComposite
from dwave.system import DWaveSampler, EmbeddingComposite, LazyFixedEmbeddingComposite
from dwave.embedding.chain_breaks import weighted_random


def draw_qac(output_name, qac_graph: PegasusK4NQACGraph, results: dimod.SampleSet, bqm: dimod.BQM, embedding):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 12))

    # regenerate the embedded variables list
    varslist = []
    for v in results.variables:
        varslist += list(embedding[v])
    emb_variables = Variables(varslist)
    nodecols = []
    nodelist = []
    mean_err_p = np.mean(results.record['errors'], axis=0)
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
    # for u, v in g2.edges:
    #     if instance.has_edge(u, v):
    #         if 'weight' in instance.edges[u, v]:
    #             w = instance.edges[u, v]['weight']
    #             if w != 0.0:
    #                 cols.append(instance.edges[u, v]['weight'])
    #                 edgelist.append((l2n[u], l2n[v]))
    # qac_graph.draw(edge_cmap=plt.cm.bwr, node_size=25, alpha=0.6, width=2.0,
    #                edge_color=cols, edgelist=edgelist, edge_vmin=-3.0, edge_vmax=3.0)
    #
    # for u, v, em in qac_graph.g.edges.data('embedding'):
    #     w = len(em)
    #     cols.append(w)
    #     edgelist.append((u, v))
    # nodecols = []
    # for u, em in qac_graph.g.nodes.data('embedding'):
    #     w = len(em)
    #     nodecols.append(w)
    # qac_graph.draw(edge_cmap=plt.cm.get_cmap('coolwarm_r'), node_size=25, alpha=0.8, width=1.2,
    #              edge_color=cols, node_color=nodecols, cmap=plt.cm.get_cmap('bwr_r'), vmin=3, vmax=6,
    #              edgelist=edgelist, edge_vmin=-3.0, edge_vmax=8.0)

    # plt.show()
    plt.savefig(output_name+"_embedding.pdf")


def main(args=None):
    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    add_qac_arguments(parser)
    parser.add_argument("--qac-mapping", type=str, default=None,
                        help="Topology mapping to QAC graph")
    args = parser.parse_args(args)

    problem_file = args.problem
    tf = args.tf
    sep = ',' if args.format == 'csv' else None
    bqm = read_ising_adjacency(problem_file, 1.0, sep, args.qubo)
    bqm = dimod.BQM(bqm)  # ensure dict-based BQM
    if args.qac_mapping is not None:
        n2l, l2n = read_mapping(args.qac_mapping)
        mapping = {k: n for (k, n) in l2n.items() if k in bqm.linear}
        mapping_n2l = {n: k for (n, k) in n2l.items() if k in bqm.linear}
        bqm.relabel_variables(mapping)
    else:
        mapping_n2l = None

    dw_sampler = DWaveSampler()

    # Interpret and construct the annealing schedule
    if args.schedule is not None:
        sched = interpret_schedule(args.tf, *args.schedule)
        print(sched)
        dw_sampler.validate_anneal_schedule(sched)
    else:
        print(f"tf={args.tf}")
        sched = None
    if args.verbose:
        print(f"QAC Penalty: {args.qac_penalty}")
        print(f"QAC Problem scale: {args.qac_scale}")
    qac_args = {
        "qac_penalty_strength": args.qac_penalty,
        "qac_problem_scale": args.qac_scale,
        "qac_decoding": args.qac_mode
    }
    sched_kwags = {"anneal_schedule": sched} if sched is not None else {"annealing_time": args.tf}
    dw_kwargs = {"num_spin_reversal_transforms": 1 if args.rand_gauge else 0,
                 "num_reads": args.num_reads,
                 "auto_scale": False}
    if args.qac_method == "qac":
        qac_sampler = PegasusQACEmbedding(16, dw_sampler)
    elif args.qac_method == "k4":
        qac_graph = PegasusK4NQACGraph.from_sampler(16, dw_sampler)
        qac_sampler = PegasusNQACEmbedding(16, dw_sampler, qac_graph)
    else:
        raise RuntimeError(f"Invalid method {args.qac_method}")

    if args.minor_embed:
        sampler = EmbeddingComposite(qac_sampler, embedding_parameters={'tries': 32, 'threads': 4})
        emb_kwargs = {
            'chain_strength': args.chain_strength,
            'chain_break_method': weighted_random,
            'return_embedding': True,
        }
    else:
        qac_sampler.validate_structure(bqm)
        sampler = qac_sampler
        emb_kwargs = {}
    sampler = ScaleComposite(sampler)
    aggr_results = run_sampler(sampler, bqm, args, aggregate=False, run_gc=True, scalar=1.0/args.scale_j,
                               **emb_kwargs, **qac_args, **dw_kwargs, **sched_kwags)

    if args.minor_embed:
        for i in range(len(aggr_results)):
            if args.draw_embedding:
                emb = aggr_results[i].info['embedding_context']['embedding']
                draw_qac(args.output+f"_{i}", qac_sampler.qac_graph, aggr_results[i], bqm, emb)
            aggr_results[i]._record = rfn.drop_fields(aggr_results[i].record, drop_names=['errors', 'ties'],
                                                      usemask=False, asrecarray=True)

    all_results: dimod.SampleSet = dimod.concatenate(aggr_results)
    if mapping_n2l is not None:
        all_results.relabel_variables(mapping_n2l)
    lo = all_results.lowest()
    lo_df: pd.DataFrame = lo.to_pandas_dataframe()
    if args.qac_method == "qac" and args.qac_mode == "qac":
        print(lo_df.loc[:, ['energy', 'error_p', 'rep', 'num_occurrences']])
    else:
        print(lo_df.loc[:, ['energy', 'rep', 'num_occurrences']])
    num_gs = np.sum(lo.record.num_occurrences)
    total_reads = np.sum(all_results.record.num_occurrences)
    print(f"The lowest energy appears in {num_gs}/{total_reads} samples")
    # samps_df = df = pd.DataFrame(all_results.record.sample, columns=all_results.variables)
    num_vars = len(all_results.variables)

    df = all_results.to_pandas_dataframe()
    df_samples = df.iloc[:, :num_vars].astype("int8")
    df_properties = df.iloc[:, num_vars:]
    h5_file = args.output+".h5"
    store = pd.HDFStore(h5_file, mode='w', complevel=5)
    store.append("samples", df_samples)
    store.append("info", df_properties)
    if args.minor_embed:
        emb_dat = {"avg_chain_length": [], "max_chain_length": []}
        for i in range(len(aggr_results)):
            emb = aggr_results[i].info['embedding_context']['embedding']
            chain_lens = [len(ci) for ci in emb.values()]
            emb_dat['avg_chain_length'].append(np.mean(chain_lens))
            emb_dat['max_chain_length'].append(np.max(chain_lens))
        emb_df = pd.DataFrame(emb_dat)
        print(emb_df.describe())
        store.append("embedding_info", emb_df)
    store.close()
    #df_samples.to_hdf(h5_file, key="samples", mode='a', complevel=5, format="table")
    #df_properties.to_hdf(h5_file, key="info", mode='a', complevel=5, format="table")


if __name__ == "__main__":
    main()
