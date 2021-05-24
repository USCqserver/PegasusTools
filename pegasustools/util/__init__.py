import numpy as np

from .adj import *
from .sched import *


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
