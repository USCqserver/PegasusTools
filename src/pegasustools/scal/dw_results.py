import pickle

import numpy as np
import pandas as pd
import yaml

from pegasustools.scal import tts, EpsilonType
from pegasustools.scal.stats import eval_pgs_tts, pgs_bootstrap, reduce_mean, boots_percentile, TTSStatistics, \
    boots_overlap


class DwRes:
    def __init__(self, file, inst=None, gs_energy=None):
        self.file = file
        self.dw_res: pd.DataFrame = pd.read_hdf(file, "info")
        self.rep_groups = self.dw_res.groupby("rep")
        self._samples_cache = None
        if inst is None:
            self.gs_energy = gs_energy
        else:
            with open(inst) as f:
                inst_info = yaml.safe_load(f)
            self.gs_energy = inst_info['gs_energy']

    def num_gauges(self):
        return self.rep_groups.ngroups

    def samples_per_gauge(self):
        count_series = self.rep_groups['energy'].count()
        return np.max(count_series)

    def load_samples(self):
        if self._samples_cache is None:
            samps: pd.DataFrame = pd.read_hdf(self.file, key='samples')
            self._samples_cache = samps
        return self._samples_cache

    def free_energy_by_gauge(self, beta=1.0):
        free_energies = []
        min_energy = np.min(self.rep_groups['energy'].min())
        for _, grp in self.rep_groups:
            e = grp['energy'] - min_energy
            w = np.exp(-beta * e)
            z = np.sum(w)
            fe = np.sum(w * e) / z
            free_energies.append(min_energy + fe)
        return np.asarray(free_energies)

    def pgs_by_gauge(self, eps=0.0, reps=0.0, tol=1.0e-4):
        if not isinstance(reps, (list, np.ndarray)):
            return np.asarray(self._collect_pgs_by_gauge(eps=eps, reps=reps, tol=tol))
        else:
            pgs_by_reps = [np.asarray(self._collect_pgs_by_gauge(eps=e, reps=re, tol=tol)) for e, re in zip(eps, reps)]
            pgs_arr = np.stack(pgs_by_reps, axis=0)  # [reps, gauge, samps ]
            return pgs_arr

    def _collect_pgs_by_gauge(self, eps=0.0, reps=0.0, tol=1.0e-4):
        rep_pgs = []
        for _, grp in self.rep_groups:
            m = np.sum(grp['num_occurrences'])
            mine_rows = grp[grp['energy'] <= self.gs_energy + eps + np.abs(self.gs_energy) * reps + tol]
            n = np.sum(mine_rows['num_occurrences'])
            rep_pgs.append(n / m)
        return rep_pgs

    def epsilon_summary(self):
        """
        Summarize the epsilons of this result
        return: (mean_eps, mean_eps2)
        tuple of ndarrays of shape [reps]
        """
        mean_eps_rep = []
        mean_eps2_rep = []
        for _, grp in self.rep_groups:
            epsilons = np.asarray(grp['energy'] - self.gs_energy)
            wi = grp['num_occurrences']
            n = np.sum(wi)
            epsilons_sq = epsilons**2
            mean_eps = np.sum(epsilons * wi) / n
            mean_eps2 = np.sum(epsilons_sq * wi) / n
            mean_eps_rep.append(mean_eps)
            mean_eps2_rep.append(mean_eps2)

        mean_eps_rep = np.stack(mean_eps_rep)
        mean_eps2_rep = np.stack(mean_eps2_rep)
        return np.stack([mean_eps_rep, mean_eps2_rep], axis=1)

    def qstats(self, rng: np.random.Generator=None):
        if rng is None:
            rng = np.random.default_rng()
        _samps = self.load_samples()
        nvars = len(_samps.columns)
        df = pd.concat([_samps, self.dw_res], axis=1)
        _dfreps = df.groupby('rep')

        qs = []
        for _, rep in _dfreps:
            nsamps = len(rep)
            si = np.asarray(rep.iloc[:nsamps, :nvars])
            wi = np.asarray(rep.iloc[:, -2])
            qi = boots_overlap(si, wi, rng) / nvars
            qs.append(qi)
        qs_arr = np.stack(qs)  # [reps]
        q2 = np.mean(qs_arr ** 2, axis=-1)
        q4 = np.mean(qs_arr ** 4, axis=-1)

        return np.stack([q2, q4], axis=1)

    def evaluate_eps_counts(self, normalize=True):
        #cdfs = []
        #for _, grp in self.rep_groups:
        #    df_es = grp.sort_values('energy', ascending=True)
        df_es = self.dw_res.sort_values('energy', ascending=True)
        epsilons = df_es['energy'] - self.gs_energy
        epsilons = np.around(epsilons, decimals=6)
        ue = np.asarray(np.unique(epsilons))
        df_es_grp = df_es.groupby(epsilons)
        counts = np.asarray(df_es_grp["num_occurrences"].sum())
        cdf = np.cumsum(counts)
        if normalize:
            tot = cdf[-1]
            cdf = np.asarray(cdf, dtype=float) / tot

        assert len(cdf) == len(ue)
        return ue, cdf
        #cdfs.append((counts, ue))
        #return cdfs

    def evaluate_eps_counts_by_rep(self, normalize=True):
        cdfs = []
        for _, grp in self.rep_groups:
            df_es = grp.sort_values('energy', ascending=True)
            epsilons = df_es['energy'] - self.gs_energy
            epsilons = np.around(epsilons, decimals=6)
            ue = np.asarray(np.unique(epsilons))
            df_es_grp = df_es.groupby(epsilons)
            counts = np.asarray(df_es_grp["num_occurrences"].sum())
            cdf = np.cumsum(counts)
            if normalize:
                tot = cdf[-1]
                cdf = np.asarray(cdf, dtype=float) / tot

            assert len(cdf) == len(ue)
            cdfs.append((ue, cdf))
        return cdfs

    def error_p_by_gauge(self):
        return self.rep_groups['error_p'].mean()


def dw_results_iter(file_template, l_list, tf_list, idx_list, gs_energies):
    if len(l_list) != len(tf_list):
        raise ValueError("Expected L list and tf list to be the same length")

    for i, (l, tf) in enumerate(zip(l_list, tf_list)):
        for k, n in enumerate(idx_list):
            try:
                dw_res = DwRes(file_template.format(l=l, tf=tf, n=n), gs_energy=gs_energies[i, k])
                yield dw_res
            except FileNotFoundError as e:
                print(e)
                yield None


def read_dw_results(file_template, eps_r_list, l_list, tf_list, idx_list, gauges,
                    gs_energies, err_p=False):
    """

    :param file_template:  Formattable string including {l}, {n}, and {tf}
    :param eps_r_list:  List of eps_r values to evaluate TTE
    :param l_list:    List of system sizes
    :param tf_list:   List of anneal times
    :param idx_list:   Indices of instances
    :param gauges:   Number of gauges per instance
    :param gs_energies: [L, Instances] array of ground state energies
    :param err_p: If the run is QAC, also return the probability that a logical qubit
            was broken
    :return:
    """

    pgs_arr = np.zeros((len(eps_r_list), len(l_list), len(tf_list), len(idx_list), gauges))
    errp_arr = np.zeros((len(l_list), len(tf_list), len(idx_list), gauges)) if err_p else None

    for i, l in enumerate(l_list):
        for j, tf in enumerate(tf_list):
            for k, n in enumerate(idx_list):
                filestr=file_template.format(l=l, tf=tf, n=n)
                try:
                    dw_res = DwRes(filestr, gs_energy=gs_energies[i, k])
                except (FileNotFoundError, KeyError) as e:
                    print(f" ** Failed to read DwRes from {filestr}")
                    print(e)
                    continue
                pgs_arr[:, i, j, k, :] = dw_res.pgs_by_gauge(reps=eps_r_list, eps=[0.0]*len(eps_r_list))
                if err_p:
                    errp_arr[i, j, k, :] = dw_res.error_p_by_gauge()

    if err_p:
        return pgs_arr, errp_arr
    else:
        return pgs_arr



def read_dw_results2(file_template, rhos_list, l_list, tf_list, idx_list, gauges,
                    gs_energies, err_p=False, fmt_kwargs=None):
    """

    :param file_template:  Formattable string including {l}, {n}, and {tf}
    :param eps_r_list:  List of eps_r values to evaluate TTE
    :param l_list:    List of system sizes
    :param tf_list:   List of anneal times
    :param idx_list:   Indices of instances
    :param gauges:   Number of gauges per instance
    :param gs_energies: [L, Instances] array of ground state energies
    :param err_p: If the run is QAC, also return the probability that a logical qubit
            was broken
    :return:
    """
    if fmt_kwargs is None:
        fmt_kwargs = {}

    pgs_arr = np.zeros((len(rhos_list), len(l_list), len(tf_list), len(idx_list), gauges))
    errp_arr = np.zeros((len(l_list), len(tf_list), len(idx_list), gauges)) if err_p else None
    for i, l in enumerate(l_list):
        instance_size=None
        for j, tf in enumerate(tf_list):
            for k, n in enumerate(idx_list):
                filestr=file_template.format(l=l, tf=tf, n=n, *fmt_kwargs)
                try:
                    dw_res = DwRes(filestr, gs_energy=gs_energies[i, k])
                except (FileNotFoundError, KeyError) as e:
                    print(f" ** Failed to read DwRes from {filestr}")
                    print(e)
                    continue
                if instance_size is None:
                    instance_size = len(dw_res.load_samples().columns)
                eps_arr = np.asarray(rhos_list) * instance_size;
                pgs_arr[:, i, j, k, :] = dw_res.pgs_by_gauge(eps=eps_arr, reps=[0.0]*len(eps_arr))
                if err_p:
                    errp_arr[i, j, k, :] = dw_res.error_p_by_gauge()

    if err_p:
        return pgs_arr, errp_arr
    else:
        return pgs_arr


def read_dw_results3(file_template, eps_list, l_list, tf_list, idx_list, gauges,
                     gs_energies, err_p=False, eps_stats=False, q_stats=False,
                     epsilon_type=EpsilonType.ABSOLUTE, fmt_kwargs=None, rng: np.random.Generator = None):
    """

    :param file_template:  Formattable string including {l}, {n}, and {tf}
    :param eps_r_list:  List of eps_r values to evaluate TTE
    :param l_list:    List of system sizes
    :param tf_list:   List of anneal times
    :param idx_list:   Indices of instances
    :param gauges:   Number of gauges per instance
    :param gs_energies: [L, Instances] array of ground state energies
    :param err_p: If the run is QAC, also return the probability that a logical qubit
            was broken
    :param rng: Random number generator. Defaults to np.random.default_rng()
        Only advanced if eps_stats or q_stats is set
    :return:

    """

    if fmt_kwargs is None:
        fmt_kwargs = {}
    pgs_arr = np.zeros((len(eps_list), len(l_list), len(tf_list), len(idx_list), gauges))
    errp_arr = np.zeros((len(l_list), len(tf_list), len(idx_list), gauges)) if err_p else None
    q_stats_arr = np.zeros((len(l_list), len(tf_list), len(idx_list), gauges, 2)) if q_stats else None
    eps_stats_arr = np.zeros((len(l_list), len(tf_list), len(idx_list), gauges, 2)) if eps_stats else None

    for i, l in enumerate(l_list):
        instance_size = None
        for j, tf in enumerate(tf_list):
            for k, n in enumerate(idx_list):
                filestr=file_template.format(l=l, tf=tf, n=n, **fmt_kwargs)
                try:
                    dw_res = DwRes(filestr, gs_energy=gs_energies[i, k])
                except (FileNotFoundError, KeyError) as e:
                    print(f" ** Failed to read DwRes from {filestr}")
                    print(e)
                    continue
                if epsilon_type == EpsilonType.RESIDUAL and instance_size is None:
                    instance_size = len(dw_res.load_samples().columns)
                if epsilon_type == EpsilonType.RESIDUAL:
                    eps_arr = np.asarray(eps_list) * instance_size
                    reps = [0.0] * len(eps_list)
                elif epsilon_type == EpsilonType.RELATIVE:
                    reps = np.asarray(eps_list)
                    eps_arr = [0.0] * len(eps_list)
                else:
                    eps_arr = np.asarray(eps_list)
                    reps = [0.0] * len(eps_list)
                pgs_arr[:, i, j, k, :] = dw_res.pgs_by_gauge(eps=eps_arr, reps=reps)
                if err_p:
                    errp_arr[i, j, k, :] = dw_res.error_p_by_gauge()
                if q_stats:
                    q_stats_arr[i, j, k, :, :] = dw_res.qstats(rng)
                if eps_stats:
                    eps_stats_arr[i, j, k, :, :] = dw_res.epsilon_summary()

    return pgs_arr, errp_arr, eps_stats_arr, q_stats_arr


def read_sa_results(file_templates, eps_list, l_list, tf_list, idx_list,
                    gs_energies, relative_eps=False, tol=1.0e-4, fmt_kwargs=None):
    """

    :param file_template:  tuple of formattables string including {l}, {n}, and {tf}
    :param eps_r_list:  List of eps_r values to evaluate TTE
    :param l_list:    List of system sizes
    :param tf_list:   List of anneal times
    :param idx_list:   Indices of instances
    :param gs_energies: [L, Instances] array of ground state energies
    :return:
    """

    if fmt_kwargs is None:
        fmt_kwargs = {}

    pgs_arr = np.zeros((len(eps_list), len(l_list), len(tf_list), len(idx_list)))
    timing_arr = np.zeros((len(l_list), len(tf_list), len(idx_list)))
    eps_arr = np.asarray(eps_list)
    for i, l in enumerate(l_list):
        instance_size = None
        for j, tf in enumerate(tf_list):
            for k, n in enumerate(idx_list):
                min_filestr = file_templates[0].format(l=l, tf=tf, n=n, **fmt_kwargs)
                samps_filestr = file_templates[1].format(l=l, tf=tf, n=n, **fmt_kwargs)

                try:
                    with open(min_filestr) as f:
                        min_results = pickle.load(f)
                except (FileNotFoundError, KeyError) as e:
                    print(f" ** Failed to read SA results from {min_filestr}")
                    print(e)
                    continue
                gs_energy = gs_energies[i, k]
                energies = min_results['energies']
                timing = min_results['timing']

                if relative_eps and instance_size is None:
                    with open(samps_filestr) as f:
                        samp_results = pickle.load(f)
                    instance_size = samp_results['instance_size']
                    del samp_results

                if relative_eps:
                    tgt_eps_arr = eps_arr * instance_size
                else:
                    tgt_eps_arr = eps_arr
                num_replicas = len(energies)
                # reshape to [num_replicas, num_epsilons]
                reached_energy = (energies[:, np.newaxis] <= gs_energy[:, np.newaxis] + tgt_eps_arr[np.newaxis, :] + tol)
                pgs = np.sum(reached_energy, axis=0) / num_replicas
                pgs_arr[:, i, j, k] = pgs[:]
                timing_arr[i, j, k] = timing

    return pgs_arr, timing_arr


class DWSuccessProbs:
    def __init__(self, pgs_arr, tflist, samps_per_gauge):
        # Raw success probabilities
        self.pgs_arr = pgs_arr  # [..., num_tf, num_instances, gauges]
        self.tflist = tflist
        self.samps_per_gauge = samps_per_gauge

    def log_tts_bboots(self, nboots):
        """
        Bayesian-bootstrapped samples of log TTS
        :return:  [..., num_tf, num_boots, num_instances] TTS array
        """
        _log_tts_boots = np.log10(
            tts(pgs_bootstrap(self.pgs_arr, self.samps_per_gauge, nboots),
                np.reshape(self.tflist, [-1, 1, 1]))
        )
        _log_tts_boots = np.swapaxes(_log_tts_boots, -2, -1)

        return _log_tts_boots

    def log_tts_percentile(self, nboots, q=0.5):
        _log_tts_boots = self.log_tts_bboots(nboots)
        log_med, log_med_err, log_med_inf = reduce_mean(
            boots_percentile(_log_tts_boots, q),
            ignore_inf=True
        )
        return TTSStatistics(log_med, log_med_err, log_med_inf)


class DWaveInstanceResults:
    """
    Contains and evaluates the D-Wave results of an entire problem instance class.

    """
    def __init__(self, path_fmt, gs_energies, llist, tflist, idxlist,
                 gauges=None, samps_per_gauge=None,
                 epsilons=None, epsilon_type: EpsilonType=EpsilonType.ABSOLUTE,
                 qac=False, eps_stats=False,
                 q_stats=False, fmt_kwargs=None):
        if fmt_kwargs is None:
            fmt_kwargs = {}
        self.path_fmt = path_fmt
        self.llist = llist
        self.tflist = tflist
        self.idxlist = idxlist
        # try to automatically determine number of gauges or samples per gauge
        if gauges is None or samps_per_gauge is None:
            filestr = path_fmt.format(l=llist[0], tf=tflist[0], n=idxlist[0], **fmt_kwargs)
            dw_res = DwRes(filestr, gs_energy=gs_energies[0, 0])
            if gauges is None:
                gauges = dw_res.num_gauges()
            if samps_per_gauge is None:
                samps_per_gauge = dw_res.samples_per_gauge()
        self.gauges = gauges
        self.samps_per_gauge = samps_per_gauge
        self.gs_energies = gs_energies
        # Raw success probabilities
        self.success_probs = None  # [num_epsilons, num_L, num_tf, num_instances, gauges]
        # Median TTSStatistics
        self.tts_array = None
        self.qac_error_probs = None
        self.qac = qac
        self.eps_stats = eps_stats
        self.q_stats = q_stats
        self.eps_stats_arr = None
        self.q_stats_arr = None
        self.fmt_kwargs = fmt_kwargs

        self.relative = epsilon_type
        if epsilons is None:
            self.epsilons = np.asarray([0.0])
        else:
            self.epsilons = np.asarray(epsilons)

    def load(self, rng: np.random.Generator = None):
        if self.success_probs is None:
            _read_dwres = read_dw_results3(self.path_fmt, self.epsilons, self.llist, self.tflist,
                                           self.idxlist, self.gauges, self.gs_energies, self.qac,
                                           epsilon_type=self.relative, rng=rng,
                                           eps_stats=self.eps_stats, q_stats=self.q_stats,
                                           fmt_kwargs=self.fmt_kwargs)
            self.success_probs, self.qac_error_probs, self.eps_stats_arr, self.q_stats_arr = _read_dwres

            # self.tts_array = eval_pgs_tts(self.success_probs, self.tflist, self.samps_per_gauge, nboots=self.nboots)

