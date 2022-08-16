import abc
import typing

import dimod
import networkx as nx
from dimod.typing import Variable
from dimod import Structured


def child_structure_dfs(sampler, seen=None) -> dimod.Structured:
    """Return the first available structured sampler using a depth-first search on its
    children.

    Args:
        sampler (:obj:`.Sampler`):
            :class:`.Structured` or composed sampler with at least
            one structured child.

        seen (set, optional, default=False):
            IDs of already checked child samplers.

    Returns:
        a structured sampler

    Raises:
        ValueError: If no structured sampler is found.

    Examples:

    >>> sampler = dimod.TrackingComposite(
    ...                 dimod.StructureComposite(
    ...                 dimod.ExactSolver(), [0, 1], [(0, 1)]))
    >>> print(dimod.child_structure_dfs(sampler).nodelist)
    [0, 1]


    """
    seen = set() if seen is None else seen

    if sampler not in seen:
        try:
            sampler.structure
            return sampler
        except AttributeError:
            # hasattr just tries to access anyway...
            pass

    seen.add(sampler)

    for child in getattr(sampler, 'children', ()):  # getattr handles samplers
        if child in seen:
            continue

        try:
            return child_structure_dfs(child, seen=seen)
        except ValueError:
            # tree has no child samplers
            pass

    raise ValueError("no structured sampler found")


class PgtStructured(Structured):
    """
    Overrides the methods of structured to construct nodelists and edgelists from
    a base NetworkX graph. The .to_networkx_graph() method also returns the base
    graph rather than constructing it from the nodelist and edgelist.
    This allows encoding samplers
    """

    @property
    def nodelist(self) -> typing.List[Variable]:
        """Nodes/variables allowed by the sampler."""
        G = self.networkx_graph
        if not hasattr(self, '_nodelist'):
            nodelist = list(G.nodes)
            self._nodelist = nodelist
            return nodelist
        else:
            return self._nodelist

    @property
    def edgelist(self) -> typing.List[typing.Tuple[Variable, Variable]]:
        """Edges/interactions allowed by the sampler.
        """
        G = self.networkx_graph
        if not hasattr(self, '_edgelist'):
            edgelist = list(G.edges)
            self._edgelist = edgelist
            return edgelist
        else:
            return self._edgelist

    @property
    @abc.abstractmethod
    def networkx_graph(self) -> nx.Graph:
        """
        The NetworkX graph of the structure
        :return:
        """
        pass

    def to_networkx_graph(self):
        """Convert structure to NetworkX graph format.

        Note that NetworkX must be installed for this method to work.

        Returns:
            :class:`networkx.Graph`: A NetworkX graph containing the nodes and
            edges from the sampler's structure.

        """
        return self.networkx_graph
