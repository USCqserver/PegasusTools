import dimod
import numpy as np
from dimod.core.bqm import BQM
from dimod.bqm import AdjVectorBQM
from pegasustools.util import bootstrap_apply, bayesian_bootstrap, apply_boots, BootstrapSample

class QuboTitration:
    """
    This class uses a given sampler to ``titrate'' a BQM in QUBO form

      H(n) = \sum_i (a_i + h) n_i + \sum_{ij} W_{ij} n_i n_j

    where h is a parameter that is varied

    """
    def __init__(self, quadratic: np.ndarray, variables=None):
        di = np.diag_indices_from(quadratic)
        n = quadratic.shape[0]
        if variables is None:
            variables = np.arange(0, n)
        if len(variables) != n:
            raise RuntimeError("Variable array must have the same length as the number of variables")
        self._variables = variables
        self._linear = quadratic[di]
        quadratic_d = {}
        for i in range(n):
            for j in range(i+1, n):
                quadratic_d[(variables[i], variables[j])] = quadratic[i, j] + quadratic[j, i]
        self._quadratic = quadratic_d
        linear = {v: l for v, l in zip(variables, self._linear)}
        self._qubo_bqm: BQM = AdjVectorBQM(linear, quadratic_d, dimod.BINARY)


    def titr_bqm(self, h):
        linear_arr = self._linear + h
        linear = {v: l for v, l in zip(self._variables, linear_arr)}
        bqm: BQM = AdjVectorBQM(linear, self._quadratic, dimod.BINARY)
        return bqm

    def sample_h(self, h, sampler: dimod.Sampler, titr_scale=False, titr_beta=None, **kwargs):
        linear = self._linear + h
        bqm : BQM= AdjVectorBQM(linear, self._quadratic, dimod.BINARY)

        results: dimod.SampleSet = sampler.sample(bqm, **kwargs)
        results = results.aggregate()
        titration = TitrationResult(bqm, results, beta=titr_beta)
        return titration


class TitrationResult:
    def __init__(self, bqm: BQM, results: dimod.SampleSet, beta=None):
        self.qubo_bqm = bqm
        self.qubo_results = results
        # resort variables to BQM order (necessary when the fixed variable composite is used)
        results_variables = results.variables
        var_idx = {v: i for i,v in enumerate(results_variables)}
        re_idx = np.asarray([var_idx[v] for v in bqm.variables])

        self.variables = bqm.variables
        ising_bqm : BQM = bqm.change_vartype(dimod.SPIN, inplace=False)
        results = results.change_vartype(dimod.SPIN, inplace=False)
        self.ising_bqm = ising_bqm
        self.ising_results = results

        samp: np.ndarray = results.record.sample[:, re_idx]
        energies: np.ndarray = results.record.energy
        min_energy = np.min(energies)
        n: np.ndarray = results.record.num_occurrences
        total_n = np.sum(n)
        logn = np.log(n) - np.log(total_n)
        if beta is not None: # Set importance sampling weights
            f_arr = beta*(energies - min_energy) + logn
            minf = np.min(f_arr)
            f_arr -= minf
            w_arr = np.exp(-f_arr)
            z = np.sum(w_arr)
            weights = w_arr / z
        else:
            weights = None
        self.sample_weights = weights
        # First moments
        #mean_s = np.sum(samp * weights[:, np.newaxis], axis=0)
        #mean_energy = bootstrap_apply(np.mean, energies, weights=weights)
        #mean_energy = bayesian_bootstrap(energies, observations=n)
        sisj = samp[:, :, np.newaxis] * samp[:, np.newaxis, :]
        mean_s, mean_s2, mean_energy = bayesian_bootstrap((samp, sisj, energies), observations=n, weights=weights)
        #mean_s = bootstrap_apply(lambda x: np.mean(x, axis=0), samp, weights=weights)
        # Calculate the mean-field energy
        def mean_field_energy(x):
            ms, labels = dimod.as_samples((x, self.variables))
            ldata, (irow, icol, qdata), offset \
                = self.qubo_bqm.to_numpy_vectors(variable_order=labels)
            qubo_x = (ms+1.0)/2.0
            mf_energies = qubo_x.dot(ldata) + (qubo_x[:, irow] * qubo_x[:, icol]).dot(qdata) + offset
            mf_energy = mf_energies[0]
            return mf_energy
        mf_energy = mean_s.apply(mean_field_energy)

        def cov_boots(si_boots, sisj_boots):
            return sisj_boots - si_boots[:, np.newaxis, :]*si_boots[:, :, np.newaxis]

        def eval_cov(x):
            samp = x[0]
            s_is_j = x[1]
            s_i = np.mean(samp, axis=0)
            #s_is_j = x[:, :, np.newaxis] * x[:, np.newaxis, :]
            return np.mean(s_is_j, axis=0) - s_i[np.newaxis, :]*s_i[:, np.newaxis]

        #mean_s2 = np.sum(sisj * weights[:, np.newaxis, np.newaxis], axis=0)
        #mean_s2 = bootstrap_apply(lambda x: np.mean(x, axis=0), sisj, weights=weights )
        #mean_s2 = bayesian_bootstrap(sisj, observations=n)
        # Covariance matrix
        cov = apply_boots(cov_boots, mean_s, mean_s2)
        #cov = mean_s2.mean - mean_s.mean[np.newaxis, :] * mean_s.mean[:, np.newaxis]
        #cov = bootstrap_apply(eval_cov, [samp, sisj], weights=weights)
        # Mean field energy
        #ms = dimod.as_samples(mean_s)
        # Use the slow way of calculating energy

        # Correlation energy
        dg_cor = apply_boots(lambda x,y: x-y, mf_energy, mean_energy)

        self.energies = mean_energy
        self.mf_energy = mf_energy
        self.mean_s : BootstrapSample = mean_s
        self.mean_s2 = mean_s2
        self.cov = cov
        self.dg_cor = dg_cor



