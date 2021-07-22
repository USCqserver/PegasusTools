#!/usr/bin/env python3
import argparse
import networkx as nx
import numpy as np
from pegasustools.util.graph import random_walk_loop
from pegasustools.util.adj import save_ising_instance_graph, ising_graph_to_bqm


def frustrated_loops(g: nx.Graph, m, j=-1.0, jf=1.0, min_loop=6, max_iters=10000):
    """
    Generate a frustrated loop problem over the logical graph
    :param g: Logical graph
    :param m: Number of clauses
    :param j: Coupling strength in each loop
    :return:
    """
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes)
    g2.add_edges_from(g.edges, weight=0.0)
    nodes_list = list(g.nodes)
    num_nodes = len(nodes_list)
    loops = []
    nloops = 0
    for _ in range(max_iters):
        node_idx = np.random.randint(0, num_nodes)
        node = nodes_list[node_idx]
        lp = random_walk_loop(node, g2)
        if len(lp) < min_loop:
            continue
        nloops += 1
        loops.append(lp)
        # randomly invert a coupling
        js = [j for _ in lp]
        i = np.random.randint(0, len(lp))
        js[i] = jf
        for u, v, ji in zip(lp[:-1], lp[1:], js):
            g2.edges[u, v]["weight"] += ji
        if nloops >= m:
            break
    else:
        raise ValueError(f"frustrated_loops failed to generate instance within {max_iters} iterations")
    #purge zero-edges
    zero_edges = []
    for u, v in g2.edges:
        if g2.edges[u, v]["weight"] == 0.0:
            zero_edges.append((u, v))
    g2.remove_edges_from(zero_edges)
    return g2, loops


def random_couplings(nodelist, edgelist):
    lin = {n: 0.0 for n in nodelist}
    #for n in nodelist:
    #    g.add_node(n, bias=0.0)
    qua = {}
    num_edges = len(edgelist)
    # Generate random couplings
    rand_js = np.random.randint(1, 4, [num_edges]) * (2 * np.random.randint(0, 2, [num_edges]) - 1)
    rand_js = rand_js / 6.0
    for e, j in zip(edgelist, rand_js):
        qua[e] = j
        #g.add_edge(e[0], e[1], coupling=j)

    return lin, qua


def main():
    parser = argparse.ArgumentParser(
        description="Generate a problem instance over an arbitrary graph"
    )
    parser.add_argument("--clause-density", type=float, default=1.0,
                        help="Number of clauses as a fraction of problem size (for clause-based instances)")
    parser.add_argument("--min-clause-size", type=int, default=6,
                        help="Minimum size of a clause (for clause-based instances)")
    parser.add_argument("--range", type=float, default=9.0,
                        help="Maximum weight of any single coupling/disjunction summed over all clauses.")
    parser.add_argument("--rejection-iters", type=int, default=1000,
                        help="If a generated instance must satisfy some constraint, the number of allowed attempts "
                        "to reject an instance generate a new one.")
    #parser.add_argument("--non-degenerate", action='store_true',
    #                    help="Force the planted ground state of a single clause to be non-degenerate.\n"
    #                         "\tfl: The AFM coupling strength is reduced to 0.75")
    parser.add_argument("topology", type=str, help="Text file specifying graph topology")
    parser.add_argument("instance_class", choices=["fl", "r3"],
                        help="Instance class to generate")
    parser.add_argument("dest", type=str,
                        help="Save file for the instance specification in Ising adjacency format")

    args = parser.parse_args()
    g: nx.Graph = nx.readwrite.read_adjlist(args.topology, nodetype=int)

    if args.instance_class == "fl":
        n = g.number_of_nodes()
        m = int(args.clause_density * n)
        print(f" * instance size = {n}")
        print(f" * clause density = {args.clause_density}")
        print(f" * num clauses = {m}")
        r = args.min_clause_size
        for i in range(args.rejection_iters):
            g2, loops = frustrated_loops(g, m, min_loop=r)
            if all(abs(j) <= args.range for (u, v, j) in g2.edges.data("weight")):
                print(f" * range {args.range} satisfied in {i} iterations")
                cc = list(c for c in nx.connected_components(g2) if len(c) > 1)
                cc.sort(key=lambda c: len(c), reverse=True)
                ccn = [len(c) for c in cc]
                print(f" * Connected component sizes: {ccn}")
                if len(ccn) > 1:
                    print(" ** Rejected multiple connected components")
                    continue
                else:
                    break
        else:
            raise RuntimeError(f"Failed to satisfy range ({args.range}) within {args.rejection_iters} iterations")
        loop_lengths = [len(l) for l in loops]
        bins = np.concatenate([np.arange(r, 4*r)-0.5, [1000.0]])
        hist, _ = np.histogram(loop_lengths, bins)
        print(bins)
        print(hist)

        bqm = ising_graph_to_bqm(g2)
        numvar = bqm.num_variables
        e = bqm.energy((np.ones(numvar, dtype=int), bqm.variables))
        print(f"* GS Energy: {e}")
        save_ising_instance_graph(g2, args.dest)


if __name__ == "__main__":
    main()
#
# bqm = AdjVectorBQM(lin, qua, dimod.SPIN)
#
# tflist = [1.0, 5.0, 25.0]
# samples_qac = [
#     qac_sampler.sample(bqm, encoding='qac', qac_penalty_strength=0.3, qac_problem_scale=1.0, num_reads=20,
#                        auto_scale=False, annealing_time=tf)
#     for tf in tflist
# ]
# samples_c = [
#     qac_sampler.sample(bqm, encoding='all', num_reads=20, auto_scale=False, annealing_time=tf)
#     for tf in tflist
# ]
# print("QAC")
# for tf, s in zip(tflist, samples_qac):
#     print(f"tf = {tf}")
#     print(s.truncate(10))
# print("ALL")
# for tf, s in zip(tflist, samples_c):
#     print(f"tf = {tf}")
#     print(s.truncate(10))