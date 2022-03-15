#!/usr/bin/env python3
import argparse
import networkx as nx
import numpy as np
import json
import yaml
from numpy.random import default_rng, Generator
from pegasustools.util.graph import random_walk_loop, random_walk_chain
from pegasustools.util.adj import save_ising_instance_graph, ising_graph_to_bqm


def frustrated_loops(g: nx.Graph, m, j=-1.0, jf=1.0, min_loop=6, max_iters=10000, rng: Generator=None):
    """
    Generate a frustrated loop problem over the logical graph
    :param g: Logical graph
    :param m: Number of clauses
    :param j: Coupling strength in each loop
    :return:
    """
    if rng is None:
        rng = default_rng()
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes)
    g2.add_edges_from(g.edges, weight=0.0)
    nodes_list = list(g.nodes)
    num_nodes = len(nodes_list)
    rand_cache_size = 10*m
    rand_ints = rng.integers(0, num_nodes, rand_cache_size)
    loops = []
    nloops = 0
    rand_idx = 0
    for i in range(max_iters):
        if rand_idx >= rand_cache_size:
            rand_ints = rng.integers(0, num_nodes, rand_cache_size)
            rand_idx = 0
        node_idx = rand_ints[rand_idx]
        rand_idx += 1

        node = nodes_list[node_idx]
        lp = random_walk_loop(node, g2, rng=rng)
        if len(lp) < min_loop:
            continue
        nloops += 1
        loops.append(lp)
        # randomly invert a coupling
        js = [j for _ in lp]
        i = rng.integers(0, len(lp))
        js[i] = jf
        # Add to the weights of the entire loop
        for u, v, ji in zip(lp[:-1], lp[1:], js):
            g2.edges[u, v]["weight"] += ji
        g2.edges[lp[-1], lp[0]]["weight"] += js[-1]
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


def random_1d_chain(g: nx.Graph, n, j=-1.0,  max_iters=10000,  rng: Generator=None):
    """
    Embed a 1D chain on a logical graph using a random walk
    :param g: Logical graph
    :param n: Chain length
    :param j: Chain Strength. Defaults to 1.0
    :param max_iters:
    :param rng:
    :return:
    """
    if rng is None:
        rng = default_rng()
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes)
    g2.add_edges_from(g.edges, weight=0.0)

    if np.isscalar(j):
        j_array = np.full(n, j)
    else:
        j_array = j

    num_nodes = g.number_of_nodes()
    init_node = rng.integers(0, num_nodes)

    for i in range(max_iters):
        rand_walk = random_walk_chain(init_node, n, g2)
        if rand_walk is not None:
            break
    else:
        raise RuntimeError(f"random_1d_chain failed to generate instance within {max_iters} iterations")
    # Add to the weights of the entire loop
    for u, v, ji in zip(rand_walk[:-1], rand_walk[1:], j_array):
        g2.edges[u, v]["weight"] += ji
    # purge zero-edges
    zero_edges = []
    for u, v in g2.edges:
        if g2.edges[u, v]["weight"] == 0.0:
            zero_edges.append((u, v))
    g2.remove_edges_from(zero_edges)

    return g2


def wishart_planted(n, m, rng: Generator=None):
    """
    Generate a Wishart planted instance on a complete K_n graph
    as described in [Hamze et. al. Phys. Rev. E 101, 052102 (2020)]
    :param n:
    :param m:
    :param rng:
    :return:
    """
    sqrt_sigma = np.sqrt(n / (n-1.0)) * (np.eye(n) - np.ones((n, n))/n)

    z = rng.normal(0.0, 1.0, (n, m))
    w_mat = sqrt_sigma @ z  # [n, m] W matrix
    j_mat = - (1.0 / n) * (w_mat @ (w_mat.T))
    tr_j = np.trace(j_mat)
    for i in range(n):
        j_mat[i, i] = 0.0

    gs_energy = 0.5 * tr_j
    g = nx.complete_graph(n)
    for u, v in g.edges:
        g.edges[u, v]['weight'] = j_mat[u, v]

    return g, gs_energy


def random_spin_glass(g: nx.Graph, coupling_set, rng: Generator):
    coupling_set = np.asarray(coupling_set)
    k = len(coupling_set)
    num_edges = g.number_of_edges()
    rand_js = rng.choice(coupling_set, num_edges)
    if rng is None:
        rng = default_rng()
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes)
    g2.add_edges_from(g.edges, weight=0.0)
    rand_js = rand_js * (2 * rng.integers(0, 2, [num_edges]) - 1)
    for i, (u, v) in enumerate(g2.edges):
        g2.edges[u, v]['weight'] = rand_js[i]

    return g2


def sidon_28(g: nx.Graph, rng: Generator=None):
    couplings = np.asarray([8, 13, 19, 28])
    return random_spin_glass(g, couplings, rng)


def random_couplings(g: nx.Graph, rng: Generator=None):
    num_edges = g.number_of_edges()
    if rng is None:
        rng = default_rng()
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes)
    g2.add_edges_from(g.edges, weight=0.0)
    rand_js = rng.integers(1, 4, [num_edges]) * (2 * rng.integers(0, 2, [num_edges]) - 1)
    # rand_js = rand_js / 6.0
    for i, (u, v) in enumerate(g2.edges):
        g2.edges[u, v]['weight'] = rand_js[i]

    return g2


def binomial_spin_glass(g: nx.Graph, rng: Generator=None):
    num_edges = g.number_of_edges()
    if rng is None:
        rng = default_rng()
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes)
    g2.add_edges_from(g.edges, weight=0.0)
    rand_js = (2.0 * rng.integers(0, 2, [num_edges]) - 1)
    for i, (u, v) in enumerate(g2.edges):
        g2.edges[u, v]['weight'] = rand_js[i]

    return g2


def dilute_bonds(g: nx.Graph, p, rng: Generator=None):
    num_edges = g.number_of_edges()
    if rng is None:
        rng = default_rng()
    rand_eps = rng.binomial(1, p, [num_edges])

    zero_edges = []
    for eps, (u, v) in zip(rand_eps, g.edges):
        if eps == 0:
            zero_edges.append((u, v))

    g.remove_edges_from(zero_edges)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def dilute_nodes(g: nx.Graph, p, rng: Generator=None):
    num_nodes = g.number_of_nodes()
    if rng is None:
        rng = default_rng()
    rand_eps = rng.binomial(1, p, [num_nodes])

    del_nodes = []
    for eps, n in zip(rand_eps, g.nodes):
        if eps == 0:
            del_nodes.append(n)
    g.remove_nodes_from(del_nodes)
    return g


def generate_wishart_planted(g: nx.Graph, args):

    m = int(args.clause_density * n)
    print(f" * instance size = {n}")
    print(f" * clause density = {args.clause_density}")
    print(f" * num clauses = {m}")
    r = args.min_clause_size
    for i in range(args.rejection_iters):
        g2, loops = frustrated_loops(g, m, min_loop=r, rng=rng)
        if all(abs(j) <= args.range for (u, v, j) in g2.edges.data("weight")):
            print(f" * range {args.range} satisfied in {i + 1} iterations")
            cc = list(c for c in nx.connected_components(g2) if len(c) > 1)
            cc.sort(key=lambda c: len(c), reverse=True)
            ccn = [len(c) for c in cc]
            print(f" * Connected component sizes: {ccn}")
            if len(ccn) > 1:
                print(" ** Rejected multiple connected components")
                continue
            else:
                e0 = 0.0
                for l in loops:
                    nl = len(l)
                    e0 += -nl + 2.0
                print(f" ** Found {len(loops)} loop instance: e_gs = {e0}")
                break
    else:
        raise RuntimeError(f"Failed to satisfy range ({args.range}) within {args.rejection_iters} iterations")
    loop_lengths = [len(l) for l in loops]
    bins = np.concatenate([np.arange(r, 4 * r) - 0.5, [1000.0]])
    hist, _ = np.histogram(loop_lengths, bins)
    print(bins)
    print(hist)

    bqm = ising_graph_to_bqm(g2)
    numvar = bqm.num_variables
    e = bqm.energy((np.ones(numvar, dtype=int), bqm.variables))
    print(f"* GS Energy: {e}")
    save_ising_instance_graph(g2, args.dest)
    if args.instance_info is not None:
        props = {"gs_energy": float(e),
                 "num_loops": len(loops),
                 "size": numvar
                 }
        with open(args.instance_info, 'w') as f:
            yaml.safe_dump(props, f, default_flow_style=False)
    return

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
    parser.add_argument("--instance-info", type=str, default=None,
                        help="Save any additional properties (e.g. ground state energy) of the generated instance "
                             "in yaml (.yml) format")
    parser.add_argument("--dilution", type=float, default=None,
                        help="Dilute the nodes by a certain probability.")
    parser.add_argument("--bond-dilution", type=float, default=None,
                        help="Dilute the bonds by a certain probability.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A manual seed for the RNG")
    parser.add_argument("-n", type=int, default=None,
                        help="Any integer index. Used in addition to the manual RNG seed if specified.")
    parser.add_argument("--coupling", type=float, default=None,
                        help="Coupling strength parameter")
    parser.add_argument("--length", type=int, default=None,
                        help="Length parameter for 1D instances")
    #parser.add_argument("--non-degenerate", action='store_true',
    #                    help="Force the planted ground state of a single clause to be non-degenerate.\n"
    #                         "\tfl: The AFM coupling strength is reduced to 0.75")
    parser.add_argument("topology", type=str, help="Text file specifying graph topology."
                                                   " Ignored if the instance class generates its own graph.")
    parser.add_argument("instance_class",
                        choices=["fl", "r3", "bsg", "wis", "r1d", "s28"],
                        help="Instance class to generate")
    parser.add_argument("dest", type=str,
                        help="Save file for the instance specification in Ising adjacency format")

    args = parser.parse_args()
    # Seed the RNG
    if args.seed is not None:
        if args.n is not None:
            seed = args.seed ^ args.n
        else:
            seed = args.seed
    else:
        seed = None
    rng = default_rng(seed)

    g: nx.Graph = nx.readwrite.read_adjlist(args.topology, nodetype=int)
    n = g.number_of_nodes()
    # FL class is treated specially right now
    if args.instance_class == "fl":
        m = int(args.clause_density * n)
        print(f" * instance size = {n}")
        print(f" * clause density = {args.clause_density}")
        print(f" * num clauses = {m}")
        r = args.min_clause_size
        for i in range(args.rejection_iters):
            g2, loops = frustrated_loops(g, m, min_loop=r, rng=rng)
            if all(abs(j) <= args.range for (u, v, j) in g2.edges.data("weight")):
                print(f" * range {args.range} satisfied in {i+1} iterations")
                cc = list(c for c in nx.connected_components(g2) if len(c) > 1)
                cc.sort(key=lambda c: len(c), reverse=True)
                ccn = [len(c) for c in cc]
                print(f" * Connected component sizes: {ccn}")
                if len(ccn) > 1:
                    print(" ** Rejected multiple connected components")
                    continue
                else:
                    e0 = 0.0
                    for l in loops:
                        nl = len(l)
                        e0 += -nl+2.0
                    print(f" ** Found {len(loops)} loop instance: e_gs = {e0}")
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
        if args.instance_info is not None:
            props = {"gs_energy": float(e),
                     "num_loops": len(loops),
                     "size": numvar
                     }
            with open(args.instance_info, 'w') as f:
                yaml.safe_dump(props, f, default_flow_style=False)
        return
    if args.instance_class == "r1d":
        # generate FM chains by default
        if args.coupling is not None:
            j = args.coupling
        else:
            j = -1.0
        if args.length is None:
            raise ValueError(f"Argument --length is required for instance class {args.instance_class}")
        n = args.length
        g2 = random_1d_chain(g, n, j, rng=rng)
        # simple ground state energy for 1D chain
        e = (n-1)*np.abs(j)
        if args.instance_info is not None:
            props = {"gs_energy": float(e),
                     "size": n
                     }
            with open(args.instance_info, 'w') as f:
                yaml.safe_dump(props, f, default_flow_style=False)
        save_ising_instance_graph(g2, args.dest)
        return

    # Generic random instances (no gs energy is known)
    if args.instance_class == "r3":
        g2 = random_couplings(g, rng)
    elif args.instance_class == "bsg":
        g2 = binomial_spin_glass(g, rng)
    elif args.instance_class == "s28":
        g2 = sidon_28(g, rng)
    else:
        raise RuntimeError(f"Instance Class {args.instance_class} is not known")
    # Apply dilution if requested
    if args.dilution is not None:
        g2 = dilute_nodes(g2, args.dilution, rng)
    if args.bond_dilution is not None:
        g2 = dilute_bonds(g2, args.bond_dilution, rng)

    save_ising_instance_graph(g2, args.dest)


if __name__ == "__main__":
    main()
