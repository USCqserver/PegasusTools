# distutils: language = c++
# cython: language_level = 3
import numpy as np

def dirichlet_samp(long n_samps, double[:] rand_gaps):
    cdef double[:] sorted_rands
    cdef long i
    if n_samps == 1: # single sample edge case
        rand_gaps[0] = 0.0
        return
    rands = np.random.rand(n_samps-1)
    sorted_rands = np.sort(rands)

    rand_gaps[0] = sorted_rands[0]
    for i in range(1, n_samps-1):
        rand_gaps[i] = sorted_rands[i] - sorted_rands[i-1]
    rand_gaps[n_samps-1] = 1.0 - sorted_rands[-1]


def bayesian_boots(long n_samps, int num_boots):
    cdef double[:, :] dp

    dp = np.empty((num_boots, n_samps), dtype=np.double)
    for i in range(num_boots):
        dirichlet_samp(n_samps, dp[i, :])

    return dp

def weighed_dirichlet_samp(long[:] observations, double[:] wdp):
    #cdef double[:] summed_rands
    cdef double[:] dp
    cdef int ridx
    cdef double p
    cdef size_t i, j, n
    cdef size_t samps = observations.shape[0]
    cdef long total_n = 0
    for i in range(samps):
        total_n += observations[i]

    # Sample an unweighed Dirichlet distribution
    dp = np.empty((total_n,), dtype=np.double)
    dirichlet_samp(total_n, dp[:])

    # Sum up the Dirchlet probabilities by sample observations
    ridx = 0
    for i in range(samps):
        n = observations[i]
        p = 0.0
        for j in range(n):
            p += dp[ridx+j]
        wdp[i] = p
        ridx += n



def weighed_bayesian_boots(long[:] observations, int num_boots):
    cdef double[:, :] wdp
    cdef size_t samps = observations.shape[0]

    wdp = np.empty((num_boots, samps), dtype=np.double)
    for i in range(num_boots):
        weighed_dirichlet_samp(observations, wdp[i, :])

    return wdp
