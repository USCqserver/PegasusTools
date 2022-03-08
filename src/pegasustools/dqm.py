import dimod
import numpy as np
from dimod import ComposedSampler, DiscreteQuadraticModel, \
    BinaryQuadraticModel
from dimod.constrained import ConstrainedQuadraticModel, cqm_to_bqm, CQMToBQMInverter
from dimod import ConstrainedQuadraticModel
from dimod.generators import combinations
from typing import Callable, Dict, List, Tuple
from dimod.typing import Variable


def dqm_to_cqm(dqm: DiscreteQuadraticModel, *, relabel_func: Callable[[Variable, int], Variable] = lambda v, c: (v, c),
               ) -> Tuple['ConstrainedQuadraticModel', Dict[Variable, List[Variable]]]:
    """
    HMB: Modification of the class method ConstrainedQuadraticModel.from_discrete_quadratic_model
    Preserves the order of all variables and cases in a DQM
    ---
    Construct a constrained quadratic model from a discrete quadratic model.

        Args:
            dqm: a discrete quadratic model.

            relabel_func (optional): A function that takes two arguments, the
                variable label and the case label, and returns a new variable
                label to be used in the CQM. By default generates a 2-tuple
                `(variable, case)`.

        Returns:
            A constrained quadratic model.

        """
    cqm = ConstrainedQuadraticModel()

    objective = BinaryQuadraticModel(dimod.Vartype.BINARY)

    var_mapping = {}
    for v in dqm.variables:
        # convert v, case to a flat set of variables
        v_vars = list(relabel_func(v, case) for case in dqm.get_cases(v))
        var_mapping[v] = v_vars
        # add the one-hot constraint
        cqm.add_discrete(v_vars, label=v)

        # add to the objective
        objective.add_linear_from(zip(v_vars, dqm.get_linear(v)))

    seen = set()
    for v in dqm.variables:
        v_vars = var_mapping[v]
        seen.add(v)
        for u in dqm.adj[v]:
            if u in seen:  # only want upper-triangle
                continue

            u_vars = list(relabel_func(u, case) for case in dqm.get_cases(u))

            objective.add_quadratic_from(
                (u_vars[cu], v_vars[cv], bias)
                for (cu, cv), bias
                in dqm.get_quadratic(u, v).items()
            )

    objective.offset = dqm.offset

    cqm.set_objective(objective)

    return cqm, var_mapping


class DQMasCQM(ComposedSampler):
    """
    Converts a DQM to a constrained binary BQM and samples it using the child sampler
    In encoding:
        (todo) Any two-state variable is reduced to a single binary variable
        Multi-state variables are constrained according to the given penalty strength
    In decoding:
        (todo) Two-state variables are always decoded exactly
        Multi-state variables are decoded by random sampling in the case of ties

    """
    children = None

    def __init__(self, child_sampler):
        self.children = [child_sampler]

    def sample(self, dqm: DiscreteQuadraticModel, constraint_penalty=None, **parameters):
        if not isinstance(dqm, DiscreteQuadraticModel):
            raise NotImplemented("DQmasCBQM expects a DQM to sample")
        cqm, var_mapping = dqm_to_cqm(dqm)
        bqm, _inverter = cqm_to_bqm(cqm, lagrange_multiplier=constraint_penalty)
        # inverter is not necessary here
        sampleset: dimod.SampleSet = self.child.sample(bqm)
        #cqm.check_feasible(sampleset)
        dqm_sampleset = self._decode_to_dqm(sampleset, var_mapping, dqm)
        return dqm_sampleset

    # might need to cythonize
    def _decode_to_dqm(self, sampleset: dimod.SampleSet, var_mapping: Dict[Variable, List[Variable]],
                        dqm: DiscreteQuadraticModel):

        dqm_samp = np.zeros([len(sampleset), dqm.num_variables()], dtype=np.int32)  # dtype should be cyDiscreteQuadraticModel.case_dtype

        for i, v in enumerate(dqm.variables):
            l = []
            var_cases = var_mapping[v]
            c = len(var_cases)
            for bv in var_cases:
                l.append(sampleset.record.sample[:, sampleset.variables.index(bv)])
            # [N, C] array, where N is the number of samples and C is the number of cases
            l = np.stack(l, axis=1)

            for dqm_row, row in zip(dqm_samp, l):
                isone = row > 0
                n = np.sum(isone)
                nz = np.argwhere(row)
                if n > 1:
                    m = nz.shape[0]
                    mi = np.random.randint(0, m)
                    dqm_row[i] = nz[mi, 0]
                else:
                    dqm_row[i] = nz[0, 0]

        energies = dqm.energies((dqm_samp, dqm.variables))
        num_occurrences = sampleset.data_vectors['num_occurrences']
        return dimod.SampleSet.from_samples((dqm_samp, dqm.variables), 'DISCRETE', energy=energies, info=sampleset.info,
                                            num_occurrences=num_occurrences)

    def sample_ising(self, h, J, **parameters):
        raise NotImplemented("DQmasCBQM expects a DQM to sample")

    def sample_qubo(self, Q, **parameters):
        raise NotImplemented("DQmasCBQM expects a DQM to sample")

    @property
    def parameters(self):
        return self.child.parameters

    @property
    def properties(self):
        return self.child.properties


def test_dqm_as_cqm():
    # shamelessly based on github.com/dwave-examples/graph-coloring
    import neal
    from dimod import ExactDQMSolver
    import networkx as nx
    import matplotlib
    import matplotlib.pyplot as plt

    print("\nSetting up graph...")
    num_colors = 5
    colors = range(num_colors)

    # Make networkx graph
    G = nx.powerlaw_cluster_graph(8, 3, 0.4)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_size=50, edgecolors='k', cmap='hsv')
    plt.savefig("dqm_test_original_graph.png")
    plt.clf()

    # initial value of Lagrange parameter
    lagrange = max(colors)

    print("\nBuilding discrete model...")

    # Initialize the DQM object
    dqm = DiscreteQuadraticModel()

    # Add the variables
    for p in G.nodes:
        dqm.add_variable(num_colors, label=p)

    # Add the biases
    for v in G.nodes:
        dqm.set_linear(v, colors)
    for u, v in G.edges:
        dqm.set_quadratic(u, v, {(c, c): lagrange for c in colors})

    sa_sampler = neal.SimulatedAnnealingSampler()
    exact_solver = ExactDQMSolver()
    sampler = DQMasCQM(sa_sampler)
    sa_sampleset = sampler.sample(dqm, constraint_penalty=10.0)
    exact_sampleset = exact_solver.sample_dqm(dqm)
    print("Done")

    for nm, sampleset in zip(["sa", "exact"], [sa_sampleset, exact_sampleset]):
        # get the first solution, and print it
        sample = exact_sampleset.first.sample
        node_colors = [sample[i] for i in G.nodes()]
        nx.draw(G, pos=pos, node_color=node_colors, node_size=50, edgecolors='k', cmap='Accent')
        plt.savefig(f'graph_result_{nm}.png')

        # check that colors are different
        valid = True
        for edge in G.edges:
            i, j = edge
            if sample[i] == sample[j]:
                valid = False
                break
        print(f"\n {nm} Solution validity: ", valid)

        colors_used = max(sample.values()) + 1
        print("\t ** Colors required:", colors_used)
