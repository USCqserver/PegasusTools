import numpy as np
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

def bayesian_bootstrap(x: Union[np.ndarray, Tuple], num_boots=32, observations: Optional[np.ndarray] = None):
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
