# distutils: language = c++
# cython: language_level = 3
import networkx as nx
from numpy.random import Generator, default_rng


def random_walk_loop(init_node, graph: nx.Graph, max_iters=1000, rng: Generator=None):
    """
    Creates a loop with a random walk.
    All nodes connected to the init_node must have a degree of at least 2
    :param init_node:
    :param graph:
    :param max_iters:
    :param rng:
    :return:
    """
    if rng is None:
        rng = default_rng()
    current_node = init_node
    next_node = None
    g = graph.copy()
    neighbors = set(g.neighbors(current_node))
    if len(neighbors) < 2:
        raise RuntimeError("random_walk_loop: Not enough initial neighbors")

    for t in range(max_iters):
        neighbors_list = list(neighbors)
        num_neighbors = len(neighbors_list)
        if num_neighbors == 0:
            raise RuntimeError(f"random_walk_loop: No neighbors available in node {current_node}")
        next_idx = rng.integers(0, num_neighbors)
        next_node = neighbors_list[next_idx]
        g.nodes[current_node]["rw_next"] = (t, next_node)
        if "rw_next" in g.nodes[next_node]:
            break
        neighbors = set(g.neighbors(next_node))
        neighbors.remove(current_node)
        current_node = next_node
    else:
        print(f"random_walk_loop timed out after {max_iters} iterations")
        return None
    # Generate the list of nodes in the self-intersecting portion of the walk
    num_nodes = g.nodes[current_node]["rw_next"][0] - g.nodes[next_node]["rw_next"][0] + 1
    init_node = next_node
    final_node = current_node
    current_node = init_node
    rand_walk = []
    for i in range(num_nodes):
        rand_walk.append(current_node)
        current_node = g.nodes[current_node]["rw_next"][1]
    assert final_node == rand_walk[-1]

    return rand_walk

