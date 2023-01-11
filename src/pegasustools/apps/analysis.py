import sys
from pathlib import Path
import logging
import argparse
from itertools import product
import pickle
from typing import List, Dict
import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import yaml

import pegasustools as pgt
from pegasustools.scal import tts
from pegasustools.scal.dw_results import read_dw_results2, DWaveInstanceResults
from pegasustools.scal.pt_results import import_pticm_dat, import_pticm_dat2, read_gs_energies, TamcPtResults, \
    read_tamc_bin, import_pticm_dat_from_pkls, eval_boots_log_tts
from pegasustools.scal.stats import boots_median, boots_percentile, pgs_bootstrap, reduce_mean, reduce_median, \
    TTSStatistics

from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
mpl.rcParams['text.usetex'] = True
rc('font',**{'family':'serif',
             'serif':['Times', 'Latin Modern Roman', 'Computer Modern Roman', 'Palatino'],
             'sans-serif'    : ['Helvetica', 'Avant Garde', 'Computer Modern Sans serif'],
             'cursive'     : ['Zapf Chancery'],
             'monospace'     : ['Courier', 'Computer Modern Typewriter'],
             'size': 12
            })
plt.rcParams['text.latex.preamble']= \
    r"""
    \usepackage{lmodern}
    \usepackage{amssymb} 
    \usepackage{amsmath}
    """
#mpl.rcParams.update({'text.latex.unicode': True})

## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

class PGTMethodBase:
    def __init__(self, name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg, rhos=None):
        self.name = name
        self.directory = method_cfg['directory']
        self.file_pattern = method_cfg['file_pattern']
        self.offset = method_cfg.get('offset', 0.0)
        self.llist = llist
        self.idxlist = idxlist
        self.gs_energies = gs_energies
        self.rhos = rhos
        self.rho_or_eps = rho_or_eps
        self.relative_epsilon = relative
        self.global_cfg = global_cfg

    def plot_tts_analysis(self, tts_statistics: TTSStatistics, instance_sizes, scaling_start=-5, ax=None,
                          plot_epsilons=None):
        if plot_epsilons is None:
            plot_epsilons = list(range(len(self.rho_or_eps)))
        loglinres= [stats.linregress(np.log10(instance_sizes[scaling_start:]),
                                      tts_statistics.mean[i, scaling_start:])
                     for i in plot_epsilons]
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()
        for i in plot_epsilons:
            eps = self.rho_or_eps[i]
            if self.relative_epsilon:
                label = f"$\\rho={self.rhos[i] * 100:3.2f}\%$"
            else:
                label = f"$\\varepsilon={eps}$"
            x = np.log10(instance_sizes)
            y, y_err = tts_statistics.mean[i, :], tts_statistics.err[i, :]
            if tts_statistics.inf_frac is not None:
                msk = tts_statistics.inf_frac[i] < 0.1
                x = x[msk]
                y = y[msk]
                y_err = y_err[msk]
            color = next(ax._get_lines.prop_cycler)['color']
            plt.errorbar(x, y, yerr=2 * y_err, color=color, ls='None',
                         label=label, capsize=4)

            x = np.log10(np.linspace((instance_sizes[scaling_start] + instance_sizes[scaling_start - 1]) / 2.0,
                                     instance_sizes[-1], 10))
            # y = pticm_pow_law(x, *pticm_powres[i][0])
            y = loglinres[i].intercept + x * loglinres[i].slope
            _k = loglinres[i].slope
            _c = loglinres[i].intercept
            plt.plot(x, y, label=f"$k={_k:3.2f}$", color=color, linestyle='--')

        return ax, loglinres


class DWMethod(PGTMethodBase):
    def __init__(self, name, method_cfg: dict, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg,
                 rhos=None):
        super(DWMethod, self).__init__(name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg,
                                       rhos=rhos)

        method_cfg.pop('directory')  # self.directory
        method_cfg.pop('file_pattern')  # self.file_pattern
        if 'offset' in method_cfg:
            method_cfg.pop('offset')
        self.annealing_times = method_cfg.pop('annealing_times')
        self.hyperparams = method_cfg
        self.num_hp = len(self.hyperparams)
        out_dir = Path(global_cfg['out_dir'])
        self.out_dir = out_dir
        tts_out_file = out_dir / f'samp_{self.name}_tts_dat.pkl'
        hparam_names = list(self.hyperparams.keys())
        hparam_iters = list(self.hyperparams.values())
        hparam_lens = list(len(v) for v in self.hyperparams.values())
        self.hparam_shape = hparam_lens
        self.pgs_shape = None
        if tts_out_file.is_file() and not global_cfg['overwrite']:
            logging.info(f"Loading sampler TTS data from {tts_out_file}.")
            with open(tts_out_file, 'rb') as f:
                hparam_array = pickle.load(f)
        else:
            hparam_array = np.ndarray(shape=hparam_lens, dtype=object)
            for idxs, HP in zip(product(*[range(n) for n in hparam_lens]), product(*hparam_iters)):
                hp_dict = {k: v for (k, v) in zip(hparam_names, HP)}
                logging.info(f"{HP}")
                dw_hp_results = DWaveInstanceResults(
                    self.directory+self.file_pattern, self.gs_energies,
                    self.llist, self.annealing_times, idxlist=self.idxlist,
                    gauges=None, samps_per_gauge=None,
                    epsilons=self.rho_or_eps, relative=self.relative_epsilon,
                    qac=False, fmt_kwargs=hp_dict)
                dw_hp_results.load()
                hparam_array[idxs] = dw_hp_results
            with open(tts_out_file, 'wb') as f:
                print(f"Saving to {tts_out_file}")
                pickle.dump(hparam_array, f)
        # tts_shape = (len(self.rho_or_eps), len(self.llist), len(self.annealing_times), len(self.idxlist), gauges)
        self.tts_shape = hparam_array.flat[0].success_probs.shape
        self.samps_per_gauge = hparam_array.flat[0].samps_per_gauge
        self.gauges = hparam_array.flat[0].gauges
        self.pgs_shape = (*self.hparam_shape, *self.tts_shape)

        self.dw_tts_array = hparam_array
        self.pgs_arr = np.zeros(self.pgs_shape, dtype=float)
        for idxs, HP in zip(product(*[range(n) for n in hparam_lens]), product(*hparam_iters)):
            self.pgs_arr[idxs] = hparam_array[idxs].success_probs

        self.boots_samples = []

    def new_bootstrap_sample(self, nboots, rng=None):
        """
                Bayesian-bootstrapped samples of log TTS
                :return:  [...,nR, nL, num_tf, num_boots, num_instances] TTS array
                """
        i = len(self.boots_samples)
        out_file = self.out_dir / f'dw_{self.name}_tts_{i}_boots.npy'
        if out_file.is_file() and not self.global_cfg['overwrite']:
            _log_tts_boots = np.load(str(out_file))
        else:
            _log_tts_boots = np.log10(
                tts(pgs_bootstrap(self.pgs_arr, self.samps_per_gauge, nboots, rng=rng),
                    np.reshape(self.annealing_times, [-1, 1, 1]))
            )
            _log_tts_boots = np.swapaxes(_log_tts_boots, -2, -1)
            np.save(str(out_file), _log_tts_boots)
        self.boots_samples.append(_log_tts_boots)
        return _log_tts_boots

    def tts_quantile(self, i=None, q=0.5, rng=None):
        if i is None:
            i = -1
        log_tts_boots = self.boots_samples[i]
        nboots = log_tts_boots.shape[-2]
        log_med, log_med_err, log_med_inf = reduce_mean(
            boots_percentile(log_tts_boots, q, rng=rng),  # [..., nR, nL, num_tf, num_boots]
            ignore_inf=True)   # [..., nR, nL, num_tf]
        if self.offset is not None:
            log_med = log_med + self.offset
        return TTSStatistics(log_med, log_med_err, log_med_inf / nboots, self.llist)

    def tts_quantile_opt(self,  i=None, q=0.5, rng=None):
        # Optimize over annealing time and any other hyperparameters
        hparam_shape = self.hparam_shape
        opt_shape = (*hparam_shape, len(self.annealing_times))
        out_shape = (len(self.rho_or_eps), len(self.llist))
        tts_stats = self.tts_quantile(i, q, rng)

        opt_tts = np.zeros(out_shape)
        opt_tts_err = np.zeros(out_shape)
        opt_tts_inf_frac = np.zeros(out_shape)
        opt_tts_tf = np.zeros(out_shape)
        opt_hparams = np.zeros((*out_shape, 1 + len(hparam_shape)), dtype=int)
        for i in range(len(self.rho_or_eps)):
            for li in range(len(self.llist)):
                arr = tts_stats.mean[..., i, li, :]
                med_amin = np.ma.argmin(arr, axis=None)
                idx_amin = np.unravel_index(med_amin, arr.shape)
                opt_tts_tf[i, li] = self.annealing_times[idx_amin[-1]]
                opt_tts[i, li] = arr[idx_amin]
                opt_tts_err[i, li] = tts_stats.err[(*idx_amin[:-1], i, li, idx_amin[-1])]
                opt_tts_inf_frac[i, li] = tts_stats.inf_frac[(*idx_amin[:-1], i, li, idx_amin[-1])]
                opt_hparams[i, li] = np.asarray([*idx_amin])

        return TTSStatistics(opt_tts, opt_tts_err, opt_tts_inf_frac, self.llist, tflist=opt_tts_tf), opt_hparams

    def tts_analysis(self, instance_sizes, i=None, q=0.5, rng=None, scaling_start=-5, **kwargs):
        tts_statistics, opt_hparams = self.tts_quantile_opt(i=i, q=q, rng=rng)

        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot()
        self.plot_tts_analysis(tts_statistics, instance_sizes, scaling_start=scaling_start, ax=ax, **kwargs)

        plt.xlabel('$\\log_{10} N$')
        plt.ylabel('$\\log_{10}$ TTE ($\\mu s$)')
        plt.legend(loc='upper left', ncol=2)
        plt.savefig(self.out_dir / f"dw_{self.name}_q{q:4.3f}_tts_scaling.pdf")


class MCMethod(PGTMethodBase):
    def __init__(self, name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg, rhos=None):
        super(MCMethod, self).__init__(name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative,
                                       global_cfg, rhos=rhos)

        out_dir = Path(global_cfg['out_dir'])
        self.out_dir = out_dir
        tts_out_file = out_dir / f'{self.name}_tts_dat.pkl'

        self.tts_out_file = tts_out_file  # Save file for all TTS data
        self.tts_samp_file = out_dir / f'mc_{self.name}_tts_samp.npz'  # save file for bootstrapped TTS samples
        if tts_out_file.is_file() and not global_cfg['overwrite']:
            logging.info(f"Loading TTS data from {tts_out_file}.")
            with open(tts_out_file, 'rb') as f:
                pticm_tts_res = pickle.load(f)
        else:
            logging.info("Processing TTS Data... ")

            pticm_tts_res = {}
            for i, l in enumerate(llist):
                logging.info(f"L={l}")
                pticm_tts_res[l] = import_pticm_dat_from_pkls(
                    self.directory + self.file_pattern, idxlist, gs_energies[i, :], list(rho_or_eps),
                    absolute=not relative, tol=1.0e-4, maxsweeps=2 ** 31,
                    fmt_kwargs={'l': l})

            with open(tts_out_file, 'wb') as f:
                print(f"Saving to {tts_out_file}")
                pickle.dump(pticm_tts_res, f)

        self.tts_data = pticm_tts_res

    def opt_tts_arr(self):
        opt_tts_list = []
        for l in self.llist:
            opt_tts_list.append(self.tts_data[l]['opt_tts'][:, np.newaxis, :])  # [nR, 1, instances]

        opt_tts_arr = np.stack(opt_tts_list, axis=1)  # [nR, nL, 1, instances]
        opt_tts_arr = np.log10(opt_tts_arr)
        return opt_tts_arr

    def tts_quantile(self, nboots, q=0.5, rng: np.random.Generator = None):

        pticm_tts_statistics: TTSStatistics = eval_boots_log_tts(self.tts_data, self.llist, q=q,
                                                                     n_boots=nboots, random_state=rng)
        return pticm_tts_statistics

    def tts_analysis(self, instance_sizes, nboots, q=0.5, rng: np.random.Generator = None,
                     scaling_start=-5, **kwargs):
        tts_statistics = self.tts_quantile(nboots, q=q, rng=rng)

        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot()
        self.plot_tts_analysis(tts_statistics, instance_sizes, scaling_start=scaling_start, ax=ax, **kwargs)

        plt.xlabel('$\\log_{10} N$')
        plt.ylabel('$\\log_{10}$ TTE ($\\mu s$)')
        plt.legend(loc='upper left', ncol=2)
        plt.savefig(self.out_dir / f"{self.name}_q{q:4.3f}_tts_scaling.pdf")


class PGTScalingAnalysis:
    def __init__(self, config_file):
        with open(config_file) as f:
            cfg: dict = yaml.safe_load(f)
        self.cfg = cfg
        self.J = cfg['j']
        self.num_instances = cfg['num_instances']
        self.n_boots = cfg['n_boots']
        self.llist = cfg['llist']
        self.instance_sizes = cfg.get('instance_sizes', self.llist)
        logging.info(f"Instance size parameters: {self.llist}")

        self.idxlist = [i for i in range(self.num_instances)]
        self.out_dir = Path(cfg['out_dir'])
        self.ovewrite = cfg['overwrite']
        self.out_dir.mkdir(exist_ok=True)
        logging.info(f"Outputting to {self.out_dir}")
        if self.ovewrite:
            logging.info(" ** Will overwrite existing ** ")
        # determine if epsilons are relative or absolute
        self.relative_epsilon = False
        if 'epsilon_type' in cfg:
            if cfg['epsilon_type'] == 'relative':
                self.relative_epsilon = True
            elif cfg['epsilon_type'] == 'absolute':
                self.relative_epsilon = False
            else:
                raise ValueError(f"Unknown epsilon_type {cfg['epsilon_type']}")

        self.epsilon_arr = np.asarray(cfg['epsilons'])
        # relative epsilons are scaled by J * instance_size
        if self.relative_epsilon:
            self.rho_or_eps = self.epsilon_arr * self.J
            self.rhos = self.epsilon_arr
            logging.info(f"Using relative epsilons:\n{self.rho_or_eps}")
        else:
            self.rho_or_eps = self.epsilon_arr
            self.rhos = None
            logging.info(f"Using absolute epsilons:\n{self.rho_or_eps}")
        self.plot_epsilons = cfg.get('plot_epsilons', None)
        # Initialize RNG and bootstrap data
        self.rand_seed = cfg.get('random_seed', 0)
        if self.rand_seed is None or self.rand_seed == 0:
            logging.info("Random seed not set.")
            self.rand_seed = np.random.SeedSequence().entropy

        logging.info(f"Initialized random seed to {self.rand_seed}")
        self.rng = np.random.Generator(np.random.PCG64(self.rand_seed))
        self.mc_samplers: Dict[MCMethod] = {}
        self.dw_samplers: Dict[DWMethod] = {}

    def validate_cfg(self, cfg_dict):
        required_keys = ["num_instances", "gs_energies",
                         "out_dir", "epsilons", "llist", "j", "file_patterns", "n_boots",
                         "random_seed"]
        for k in required_keys:
            if k not in cfg_dict.keys():
                raise ValueError(f"Required key {k} not found in config file")

    def load_gs_energies(self):
        gs_energies_dir = self.cfg['gs_energies']
        # process ground state energies
        gs_energies_file = self.out_dir / 'gs_energies.npy'
        if gs_energies_file.is_file() and not self.ovewrite:
            logging.info(f"Loading GS energies from {gs_energies_file}")
            gs_energies = np.load(str(gs_energies_file))
        else:
            logging.info(f"Reading GS energies ...")
            gs_energies = read_gs_energies(gs_energies_dir + self.cfg['file_patterns']['gs_energies'],
                                           self.llist, self.idxlist)
            np.save(str(gs_energies_file), gs_energies)
        self.gs_energies = gs_energies

    def load_mc_sampler_data(self):
        # Read the MC samplers configurations
        mc_samplers: dict = self.cfg['mc_samplers']
        self.mc_samplers = {}
        for k, v in mc_samplers.items():
            self.mc_samplers[k] = MCMethod(k, v, self.gs_energies, self.llist, self.idxlist, self.rho_or_eps,
                                             self.relative_epsilon, self.cfg, rhos=self.rhos)

    def load_dw_sampler_data(self):
        # Read the quantum samplers configurations
        dw_samplers: dict = self.cfg['quantum_samplers']
        self.dw_samplers = {}
        for k, v in dw_samplers.items():
            self.dw_samplers[k] = DWMethod(k, v, self.gs_energies, self.llist, self.idxlist, self.rho_or_eps,
                                             self.relative_epsilon, self.cfg, rhos=self.rhos)

    def draw_bootstrap_samples(self):
        for s in self.dw_samplers.values():
            s.new_bootstrap_sample(self.n_boots, rng=self.rng)

    def tts_analysis(self):
        logging.info("Writing TTS analysis ... ")
        for s in self.mc_samplers.values():
            logging.info(f"{s.name}")
            s.tts_analysis(self.instance_sizes, self.n_boots, q=0.5, rng=self.rng, scaling_start=-5, plot_epsilons=self.plot_epsilons)
        for s in self.dw_samplers.values():
            logging.info(f"{s.name}")
            s.tts_analysis(self.instance_sizes, q=0.5, rng=self.rng, scaling_start=-5, plot_epsilons=self.plot_epsilons)

    def speedup_analysis(self):
        speedup_jobs = self.cfg.get('speedup_analysis', {})
        if len(speedup_jobs) > 0:
            logging.info("Writing speedup analysis")
            for k, v in speedup_jobs.items():
                name = k
                reference_name = v['reference']
                target_name = v['target']
                if reference_name in self.mc_samplers and target_name in self.dw_samplers:
                    # Evaluate the speedup of a quantum/sampler method over a monte-carlo method
                    reference = self.mc_samplers[reference_name]
                    target = self.dw_samplers[target_name]
                    ref_boots_samps = reference.opt_tts_arr()  # [nR, nL, 1, instances]
                    target_boots, target_hp = target.tts_quantile_opt(i=-1, q=0.5, rng=self.rng)
                    target_boot_samps = target.boots_samples[-1]  # [*hparams, nR, nL, tf, nboots, instances]
                    nR = len(self.rho_or_eps)
                    nL = len(self.llist)
                    target_boot_opt = np.zeros((nR, nL, *target_boot_samps.shape[-2:]))
                    for i, j in product(range(nR), range(nL)):
                        target_boot_opt[i, j] = target_boot_samps[(*target_hp[i, j, :-1],)][i, j, target_hp[i, j, -1]]  # [nR, nL, nboots, instances]

                    speedup_array = ref_boots_samps - target_boot_opt
                    target.boots_samples.append(speedup_array)
                    tts_statistics = target.tts_quantile(i=-1, q=0.5, rng=self.rng)
                    fig = plt.figure(figsize=(7, 7))
                    ax = plt.subplot()
                    _, loglinres = target.plot_tts_analysis(tts_statistics, self.instance_sizes,
                                                            scaling_start=-5, ax=ax, plot_epsilons=self.plot_epsilons)

                    plt.xlabel('$\\log_{10} N$')
                    plt.ylabel('$\\log_{10} \\mathrm{Speedup} $')
                    plt.legend(loc='upper left', ncol=2)
                    plt.savefig(self.out_dir / f"speedup_{name}_q{0.5:4.3f}.pdf")
                    target.boots_samples.pop(-1)
                else:
                    logging.warning(f"The reference-target pair ({reference_name}, {target_name}) was not recognized. "
                                    "A valid (MC, DW) or (DW, DW) pair must be specified")


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml")
    args = parser.parse_args()
    # initialize logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    # load config file
    config_file = args.config_yaml
    pgt_analysis = PGTScalingAnalysis(config_file)
    pgt_analysis.load_gs_energies()
    pgt_analysis.load_mc_sampler_data()
    pgt_analysis.load_dw_sampler_data()
    pgt_analysis.draw_bootstrap_samples()
    pgt_analysis.tts_analysis()
    pgt_analysis.speedup_analysis()


if __name__ == "__main__":
    main()
