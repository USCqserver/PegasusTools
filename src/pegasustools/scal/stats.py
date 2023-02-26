from collections import namedtuple

import numpy as np
import scipy.stats as stats

from pegasustools.scal import tts


def not_nan(x):
    return x[np.logical_not(np.isnan(x))]


def weighted_quantile(values, quantiles, sample_weight):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values_sorted = False
    old_style = False

    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def beta_samp(p, n, nsamps):
    if p <= 0.0:
        return np.full(nsamps, 0.0)
    elif p >= 1.0:
        return np.full(nsamps, 1.0)
    else:
        return stats.beta.rvs(p * n, (1.0 - p) * n, size=nsamps)


def beta_sample(p, n, nsamps, rng=None):
    """
    Sample from the beta distribution of a probability array with n measurements,
    i.e. a = p * n successful observations and b = (1.0 - p) * n unsuccessful observations
    :param p: Probability array
    :param n: Observations, broadcastable with p
    :param nsamps: Number of samples to draw
    :return:
        Array of samples from the beta distribution, shaped (*broadcast_shape(p, n), nsamps)
    """
    if rng is None:
        rng = np.random.default_rng()

    out_shape = (*p.shape, nsamps)
    p = np.clip(p, 1.0e-8/n, 1.0 - 1.0e-8/n)
    a = p * n
    b = (1.0 - p) * n
    a = a[..., np.newaxis]
    b = b[..., np.newaxis]
    samp = rng.beta(a, b, size=out_shape)
    return samp


arr_beta_samp = np.vectorize(beta_samp, signature='(),(),()->(n)')

par_weighted_quantiles = np.vectorize(weighted_quantile, signature='(n),(),(n)->()')


def boots_median(x, nboots=10):
    n = x.shape[-1]
    pre_dims = list(x.shape[:-1])
    x_resh = x[..., np.newaxis, :]
    x_boots = np.random.choice(np.arange(n), size=pre_dims + [nboots, n], replace=True)
    boots = np.take_along_axis(x_resh, x_boots, axis=-1)
    md = np.median(boots, axis=-1)
    return np.mean(md, axis=-1), np.std(md, axis=-1)


def reduce_mean(x, ignore_inf=False, axis=-1):
    # evaluate if all finite
    isfin = np.isfinite(x)
    if not ignore_inf:
        allfin = np.all(isfin, axis=axis)
        x_mean = np.mean(x, axis=axis)
        x_std = np.std(x, axis=axis)
        x_mean = np.where(allfin, x_mean, np.inf)
        x_std = np.where(allfin, x_std, 0.0)

        return x_mean, x_std
    else:
        x_mask = np.ma.array(x,mask=np.logical_not(isfin))
        x_mean = np.ma.mean(x_mask, axis=axis)
        x_std = np.ma.std(x_mask, axis=axis)
        x_inf = np.sum(x_mask.mask, axis=axis)
        return x_mean, x_std, x_inf


def reduce_median(x):
    """
    x: np.array
    returns: (mean, std)

    Dimensions of x are [..., nboots, n].
    Takes the median over the last dimension (n) then evaluates the
    mean and standard deviation over the second to last dimension (nboots).
    If any median is not finite, then returns (np.inf, 0)
    """
    md = np.median(x, axis=-1)
    isfin = np.isfinite(md)

    md = np.where(isfin, md, 0.0)
    allfin = np.all(isfin, axis=-1)
    mdmean = np.mean(md, axis=-1)
    mdstd = np.std(md, axis=-1)
    mdmean = np.where(allfin, mdmean, np.inf)
    mdstd = np.where(allfin, mdstd, 0.0)

    return mdmean, mdstd


def boots_percentile(x, p, n_boots=None, rng: np.random.Generator = None):
    """
    Evaluates the percentile of the last dimension of x with where
    the second-to-last dimension is an existing bootstrap dimension

    x: np.array with dimensions [..., n_boots, num_instances]
    returns: (mean, std) of the percentile over the last dimension
    If any percentile is not finite, then it evaluates to (np.inf, 0)
    """
    if rng is None:
        rng = np.random.default_rng()

    pre_dims = x[..., 0:1, 0:1].size
    n = x.shape[-1]
    boots_dim = x.shape[-2]
    if n_boots is not None and boots_dim == 1:
        boots_dim = n_boots

    # n_boots samples from the dirichlet distribution
    alpha = np.full(n, 1)
    dir_samp = rng.dirichlet(alpha, size=pre_dims * boots_dim)  # [size, n]
    dir_samp = np.reshape(dir_samp, list(x.shape[:-2]) + [boots_dim, n])  # [..., n_boots, n]
    perc = par_weighted_quantiles(x, p, dir_samp)  # [..., n_boots]

    return perc


def pgs_bootstrap(pgs_arr, nsamps, boot_samps, rng: np.random.Generator = None):
    """
    pgs_arr: [..., r] dimensional array
    nsamps: number of samples per gauge
    boot_samps: number of bootstrap samples

    returns: [..., boot_samps] array of bayesian bootstrapped ground state probabilities

    Caution: This bootstrapping procedure may consume a large amount of memory
    """
    if rng is None:
        rng = np.random.default_rng()
    # resample from beta_distribution
    pgs_samp = beta_sample(pgs_arr, nsamps, boot_samps, rng=rng)  # [ ..., r, boot_samps]
    pgs_samp = np.swapaxes(pgs_samp, -2, -1)  # [ ..., boot_samps, r]
    r = pgs_arr.shape[-1]
    pre_dims = pgs_arr[..., 0:1].size

    # Dirichlet dist sample
    alpha = np.full(r, 1)
    dir_samp = rng.dirichlet(alpha, size=pre_dims * boot_samps)  # [size, r]
    dir_samp = np.reshape(dir_samp, list(pgs_arr.shape[:-1]) + [boot_samps, r])  # [..., boot_samps, r]
    boots_samp = np.sum(pgs_samp * dir_samp, axis=-1)  # [..., boot_samps]

    return boots_samp


class TTSStatistics:
    def __init__(self, mean, err, inf_frac=None, l_list=None, tflist=None):
        self.mean = mean
        self.err = err
        self.inf_frac = inf_frac
        self.l_list = np.asarray(l_list)
        self.tflist = tflist

    def save_npz(self, file):
        np.savez(file, L=np.asarray(self.l_list),
                 log_tts_mean=self.mean,
                 log_tts_err=self.err)

    def __getitem__(self, idx):
        return TTSStatistics(self.mean[idx], self.err[idx], self.inf_frac[idx], self.l_list, self.tflist)

    @staticmethod
    def load_npz(file):
        dat = np.load(file)
        ttss = TTSStatistics(dat['mean'], dat['err'])
        if 'l_list' in dat:
            ttss.l_list = dat['l_list']
        if 'inf_frac' in dat:
            ttss.inf_frac = dat['inf_frac']
        if 'tf_list' in dat:
            ttss.tflist = dat['tflist']
        return ttss

#TTSStatistics = namedtuple("TTSStatistics", "median err inf_frac")

def eval_pgs_tts(pgs_array, tf_list, samps_per_gauge, nboots=200):
    log_tts_boots = np.log10(
        tts(pgs_bootstrap(pgs_array, samps_per_gauge, nboots),
        np.reshape(tf_list, [-1, 1, 1]),
        eps=0.0)
    )
    log_med, log_med_err, log_med_inf = reduce_mean(
        boots_percentile(np.swapaxes(log_tts_boots, -2, -1), 0.5),
        ignore_inf=True
    )
    return TTSStatistics(log_med, log_med_err, log_med_inf)
