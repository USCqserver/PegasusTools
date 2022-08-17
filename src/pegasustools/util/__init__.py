import numpy as np
from numpy.lib import recfunctions
from dimod import SampleSet
from typing import Union, List, Optional, Tuple
from .adj import *
from .sched import *
from .stats import weighed_bayesian_boots, bayesian_boots


def ising_to_intlabel(samples: np.ndarray):
    """
    Converts the array of samples to an array of integer labels based
    on the binary representation of the samples
        [ nnn ] = \sum_{k=0}^n  ((s_k - 1) / 2) * 2^k
    The sample dimension cannot be greater than 32
    """
    sh = samples.shape
    if len(sh) < 2:
        raise ValueError("Expected at least a 2D array")
    n_arr = np.zeros(sh[0:-1], dtype=np.uint32)
    for k in range(sh[-1]):
        s = samples[..., k]
        d = ((1 - s)//2)
        d = d.astype(np.uint32)
        d = d * 2**k
        n_arr += d.astype(np.uint32)
    return n_arr


def intlabel_to_ising(n_arr: np.ndarray, s_max):
    """

    """

    sh = n_arr.shape
    dt = n_arr.dtype
    if not np.issubdtype(dt, np.integer):
        raise ValueError("Expected integer dtype array")

    n_arr = n_arr.copy()
    samples_arr = np.zeros([*sh, s_max], dtype=np.int8)
    for k in range(s_max):
        samples_arr[..., k] = 2 * (n_arr[..., k] ^ 1) - 1
        n_arr >>= n_arr

    return samples_arr

class BootstrapSample:
    def __init__(self, samp: np.ndarray):
        self.mean = np.mean(samp, axis=0)
        self.err = np.std(samp, axis=0)
        self.nsamps = samp.shape[0]
        self.samp = samp

    def __repr__(self):
        return f"{self.mean} +/- {self.err} (n={self.nsamps})"

    def iter_samps(self):
        for si in self.samp:
            yield si

    def apply(self, f,):
        fsamp = np.asarray([f(s) for s in self.samp])
        return BootstrapSample(fsamp)

def apply_boots(f, *boots):
    samps = tuple(b.samp for b in boots)
    fs = f(*samps)
    if isinstance(fs, tuple):
        return tuple(BootstrapSample(fsi) for fsi in fs)
    else:
        return BootstrapSample(fs)


def reweighed_means(x: np.ndarray, p: np.ndarray):
    """

    :param x:  (nsamps, ...)
    :param p:  (num_boots, nsamps)
    :return:
    """
    x_sh = x.shape
    p_sh2 = list(p.shape) + ([1]*len(x_sh[1:]))
    p2 = np.reshape(p, p_sh2)  # (num_boots, nsamps, ...)
    return np.sum(x[np.newaxis, :] * p2, axis=1)


def boltzmann_dirichlet(num_boots, energies: np.ndarray, observations: Optional[np.ndarray]=None):
    n_samps = energies.shape[0]
    if observations is not None:
        probs = weighed_bayesian_boots(observations, num_boots) # (num_boots, n_samps)
    else:
        probs = bayesian_boots(n_samps, num_boots)
    f = energies[np.newaxis, :] + np.log(probs)
    minf = np.min(f, axis=1, keepdims=True)
    f_arr = f - minf
    w_arr = np.exp(-f_arr)
    z = np.sum(w_arr, axis=1, keepdims=True)
    weights = w_arr / z
    return weights


def boltzmann_dirichlet_bootstrap(x: Union[np.ndarray, Tuple], energies: np.ndarray,
                                  observations: Optional[np.ndarray] = None, num_boots=32 ):
    n_samps = energies.shape[0]

    if np.ndim(observations) != 1 and observations.shape[0] != n_samps:
        raise ValueError(f"Observations must be a 1D array with length {n_samps}")
    probs = boltzmann_dirichlet(num_boots, energies, observations)
    if isinstance(x, tuple):
        boots = tuple(BootstrapSample(reweighed_means(xi, probs)) for xi in x)
    else:
        boots = BootstrapSample(reweighed_means(x, probs))
    return boots

def bayesian_bootstrap(x: Union[np.ndarray, Tuple], num_boots=32, observations: Optional[np.ndarray] = None,
                       weights: Optional[np.ndarray] = None):
    """

    :param x:
    :return:
    """
    if isinstance(x, tuple):
        n_samps = x[0].shape[0]
    else:
        n_samps = x.shape[0]
    if observations is None:
        probs = bayesian_boots(n_samps, num_boots)
    else:
        if np.ndim(observations) != 1 and observations.shape[0] != n_samps:
            raise ValueError(f"Observations must be a 1D array with length {n_samps}")
        probs = weighed_bayesian_boots(observations, num_boots)
    if weights is not None:
        p2 = probs * weights[np.newaxis, :]
        z = np.sum(p2, axis=1, keepdims=True)
        probs = p2 / z
    if isinstance(x, tuple):
        boots = tuple(BootstrapSample(reweighed_means(xi, probs)) for xi in x)
    else:
        boots = BootstrapSample(reweighed_means(x, probs))
    return boots


def bootstrap_apply(f, x: Union[np.ndarray, List[np.ndarray]], samps=32, weights=None) -> BootstrapSample:
    """
    Apply f to bootstrap samples of x with sample dimension 0
    If x is a list of arrays, then it is interpreted as a collection of sample features
    and the arrays are sampled jointly, passing a list of arrays with the same non-batch shape as x to f.
    f must return a single number or np.ndarray
    :param f:
    :param x:
    :param samps:
    :param weights:
    :return:
    """
    if isinstance(x, list):
        n = x[0].shape[0]
    else:
        n = x.shape[0]

    if weights is None:
        idxs = np.random.random_integers(0, n, (samps, n))
    else:
        p = weights / np.sum(weights)
        idxs = np.random.choice(np.arange(0, n), (samps, n), p=p)

    if isinstance(x, list):
        fx_samp = [f([xl[idxs[s]] for xl in x]) for s in range(samps)]
    else:
        sample = x[idxs]
        fx_samp = np.asarray([f(xi) for xi in sample])

    return BootstrapSample(fx_samp)


def _iter_records(samplesets, vartype, variables):
    # coerce each record into the correct vartype and variable-order
    for samples in samplesets:

        # coerce vartype
        if samples.vartype is not vartype:
            samples = samples.change_vartype(vartype, inplace=False)

        if samples.variables != variables:
            new_record = samples.record.copy()
            order = [samples.variables.index(v) for v in variables]
            new_record.sample = samples.record.sample[:, order]
            yield new_record
        else:
            # order matches so we're done
            yield samples.record


def concatenate(samplesets, defaults=None, concat_info: Optional[Dict] = None):
    """Combine sample sets.

    Args:
        samplesets (iterable[:obj:`.SampleSet`):
            Iterable of sample sets.

        defaults (dict, optional):
            Dictionary mapping data vector names to the corresponding default values.

        concat_info: concatenated information dictionary
    Returns:

        :obj:`.SampleSet`: A sample set with the same vartype and variable order as the first
        given in `samplesets`.

    Examples:
        >>> a = dimod.SampleSet.from_samples(([-1, +1], 'ab'), dimod.SPIN, energy=-1)
        >>> b = dimod.SampleSet.from_samples(([-1, +1], 'ba'), dimod.SPIN, energy=-1)
        >>> ab = dimod.concatenate((a, b))
        >>> ab.record.sample
        array([[-1,  1],
               [ 1, -1]], dtype=int8)

    """

    itertup = iter(samplesets)

    try:
        first = next(itertup)
    except StopIteration:
        raise ValueError("samplesets must contain at least one SampleSet")

    vartype = first.vartype
    variables = first.variables

    records = [first.record]
    records.extend(_iter_records(itertup, vartype, variables))

    # dev note: I was able to get ~2x performance boost when trying to
    # implement the same functionality here by hand (I didn't know that
    # this function existed then). However I think it is better to use
    # numpy's function and rely on their testing etc. If however this becomes
    # a performance bottleneck in the future, it might be worth changing.
    record = recfunctions.stack_arrays(records, defaults=defaults,
                                       asrecarray=True, usemask=False)
    if concat_info is None:
        concat_info = {}
    return SampleSet(record, variables, concat_info, vartype)