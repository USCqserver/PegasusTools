import dimod
import operator
import numpy as np
from dwave.system.composites.cutoffcomposite import _restore_isolated


class ThermalFixComposite(dimod.FixedVariableComposite):
    def __init__(self, child_sampler, beta=1.0, p=1.0e-4):
        super().__init__(child_sampler)
        self._beta = beta
        self._p = p
        self._f = -np.log(p)

    def sample(self, bqm: dimod.BQM, **parameters):
        """
        Fix any variables that are effectively frozen at temperature beta
        :param bqm:
        :param parameters:
        :return:
        """
        vartype = bqm.vartype
        ising_bqm = bqm.spin
        variables = list(ising_bqm.variables)
        lin = np.asarray(list(ising_bqm.linear[v] for v in variables))
        max_quads = np.asarray([np.sum(np.abs(np.asarray(list(ising_bqm.adj[v].values())))) for v in variables])
        fixed_variables = {}
        lo = self._beta * (lin - max_quads)
        hi = self._beta * (lin + max_quads)
        for i,v in enumerate(variables):
            if hi[i] <= -self._f: # frozen at +1
                fixed_variables[v] = 1
                q = 1 if vartype == dimod.SPIN else 1
                print(f" * variable {v} was frozen to {q}")
            if lo[i] >= self._f: # frozen at -1
                fixed_variables[v] = -1
                q = -1 if vartype == dimod.SPIN else 0
                print(f" * variable {v} was frozen to {q}")
        if len(fixed_variables) > 0:
            print(f" ** fixed {len(fixed_variables)}/{len(variables)} variables")
        if vartype == dimod.BINARY:
            fixed_variables = {v: (fixed_variables[v]+1)//2 for v in fixed_variables.keys()}

        sample = super().sample(bqm, fixed_variables=fixed_variables, **parameters)
        return sample


class CutOffComposite(dimod.ComposedSampler):
    """Composite to remove interactions below a specified cutoff value.

    Prunes the binary quadratic model (BQM) submitted to the child sampler by
    retaining only interactions with values commensurate with the sampler's
    precision as specified by the ``cutoff`` argument. Also removes variables
    isolated post- or pre-removal of these interactions from the BQM passed
    on to the child sampler, setting these variables to values that minimize
    the original BQM's energy for the returned samples.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler.

       cutoff (number):
            Lower bound for absolute value of interactions. Interactions
            with absolute values lower than ``cutoff`` are removed. Isolated variables
            are also not passed on to the child sampler.

       cutoff_vartype (:class:`.Vartype`/str/set, default='SPIN'):
            Variable space to execute the removal in. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

       comparison (function, optional):
            A comparison operator for comparing interaction values to the cutoff
            value. Defaults to :func:`operator.lt`.

    Examples:
        This example removes one interaction, ``'ac': -0.7``, before embedding
        on a D-Wave system. Note that the lowest-energy sample for the embedded problem
        is ``{'a': 1, 'b': -1, 'c': -1}`` but with a large enough number of samples
        (here ``num_reads=1000``), the lowest-energy solution to the complete BQM is
        likely found and its energy recalculated by the composite.

        >>> import dimod
        >>> sampler = DWaveSampler(solver={'qpu': True})
        >>> bqm = dimod.BinaryQuadraticModel({'a': -1, 'b': 1, 'c': 1},
        ...                            {'ab': -0.8, 'ac': -0.7, 'bc': -1},
        ...                            0,
        ...                            dimod.SPIN)
        >>> CutOffComposite(AutoEmbeddingComposite(sampler), 0.75).sample(bqm,
        ...                 num_reads=1000).first.energy
        -3.5

    """

    @dimod.decorators.vartype_argument('cutoff_vartype')
    def __init__(self, child_sampler, cutoff, cutoff_vartype=dimod.SPIN,
                 comparison=operator.lt):
        self._children = [child_sampler]
        self._cutoff = cutoff
        self._cutoff_vartype = cutoff_vartype
        self._comparison = comparison

    @property
    def children(self):
        """List of child samplers that that are used by this composite."""
        return self._children

    @property
    def parameters(self):
        """A dict where keys are the keyword parameters accepted by the sampler methods
        and values are lists of the properties relevent to each parameter."""
        return self.child.parameters.copy()

    @property
    def properties(self):
        """A dict containing any additional information about the sampler."""
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm: dimod.BQM, **parameters):
        """Cut off interactions and sample from the provided binary quadratic model.

        Prunes the binary quadratic model (BQM) submitted to the child sampler
        by retaining only interactions with value commensurate with the
        sampler's precision as specified by the ``cutoff`` argument. Also removes
        variables isolated post- or pre-removal of these interactions from the
        BQM passed on to the child sampler, setting these variables to values
        that minimize the original BQM's energy for the returned samples.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        Examples:
            See the example in :class:`CutOffComposite`.

        """
        child = self.child
        cutoff = self._cutoff
        cutoff_vartype = self._cutoff_vartype
        comp = self._comparison

        if cutoff_vartype is dimod.SPIN:
            original = bqm.spin
        else:
            original = bqm.binary

        # remove all of the interactions less than cutoff
        new = type(bqm)(original.linear,
                        ((u, v, bias)
                         for (u, v), bias in original.quadratic.items()
                         if not comp(abs(bias), cutoff)),
                        original.offset,
                        original.vartype)

        # next we check for isolated qubits and remove them, we could do this as
        # part of the construction but the assumption is there should not be
        # a large number in the 'typical' case
        #isolated = [v for v in new if not new.adj[v]]
        isolated = [v for v in new.variables if not new.adj[v]]
        new.remove_variables_from(isolated)

        if isolated and len(new) == 0:
            # in this case all variables are isolated, so we just put one back
            # to serve as the basis
            v = isolated.pop()
            new.linear[v] = original.linear[v]

        # get the samples from the child sampler and put them into the original vartype
        sampleset = child.sample(new, **parameters).change_vartype(bqm.vartype, inplace=True)

        # we now need to add the isolated back in, in a way that minimizes
        # the energy. There are lots of ways to do this but for now we'll just
        # do one
        if isolated:
            samples, variables = _restore_isolated(sampleset, bqm, isolated)
        else:
            samples = sampleset.record.sample
            variables = sampleset.variables

        vectors = sampleset.data_vectors
        vectors.pop('energy')  # we're going to recalculate the energy anyway

        newsampleset = dimod.SampleSet.from_samples_bqm((samples, variables), bqm, info=sampleset.info,**vectors)
        return newsampleset
