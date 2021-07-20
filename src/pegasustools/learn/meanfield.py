import dimod
import numpy as np
from dimod import Sampler
from dimod.core.bqm import BQM


class MeanFieldSolver:

    def __init__(self):
        pass

    def sample(self, bqm: BQM, mf_beta=1.0, num_reads=8, mf_init=None, max_iter=1000, atol=1.0e-3):
        vartype = bqm.vartype
        variables=bqm.variables
        n = bqm.num_variables
        qubo_w: np.ndarray = bqm.to_numpy_matrix(variable_order=list(variables))
        linear = qubo_w[np.diag_indices_from(qubo_w)]
        quadratic = qubo_w*(1.0-np.eye(n))
        quadratic = (quadratic + np.transpose(quadratic))

        if mf_init is None:
            s0 = 0.5

            mf_init = np.full((num_reads,n), s0)
        else:
            if np.ndim(mf_init) != 1 or mf_init.shape[0] != n:
                raise RuntimeError(f"mf_init must be a 1D array with length {n}")

        def local_field(x):
            h = linear[np.newaxis, :] + np.dot(x, quadratic)
            #h = linear + np.matmul(quadratic, x)
            xmean = (np.tanh(-mf_beta * h/2.0)+1.0)/2.0
            return xmean

        def energy(x):
            h = linear[np.newaxis, :] + np.dot(x, quadratic)/2.0
            #h = linear + np.matmul(quadratic, x)/2.0
            return np.sum(x*h, axis=1)

        x = mf_init*(1.0 + 0.1*np.random.randn(num_reads, n))
        for _ in range(max_iter):
            x2 = local_field(x)
            if np.mean(np.sum(np.abs(x-x2), axis=1)) < atol:
                x = x2
                break
            x = x2
        else:
            print(f" ** Mean field solver did not converge after {max_iter} iterations")
        samp = dimod.as_samples((x, variables))
        e = energy(x)
        sampset = dimod.SampleSet.from_samples(samp, dimod.BINARY, e)
        sampset = sampset.change_vartype(vartype)
        return sampset



