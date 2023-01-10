import sys
from pathlib import Path
import logging
import argparse
from collections import namedtuple
from itertools import product
from multiprocessing import Pool
import pickle
import numpy as np
import pandas as pd
import yaml

import pegasustools as pgt
from pegasustools.scal import tts
from pegasustools.scal.dw_results import read_dw_results2, DWaveInstanceResults
from pegasustools.scal.pt_results import import_pticm_dat, import_pticm_dat2, read_gs_energies, TamcPtResults, \
    read_tamc_bin, import_pticm_dat_from_pkls, eval_boots_log_tts
from pegasustools.scal.stats import boots_median, boots_percentile, pgs_bootstrap, reduce_mean, reduce_median, \
    TTSStatistics


class PGTMethodBase:
    def __init__(self, name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg):
        self.name = name
        self.directory = method_cfg['directory']
        self.file_pattern = method_cfg['file_pattern']
        self.llist = llist
        self.idxlist = idxlist
        self.gs_energies = gs_energies
        self.rho_or_eps = rho_or_eps
        self.relative_epsilon = relative
        self.global_cfg = global_cfg


class DWMethod(PGTMethodBase):
    def __init__(self, name, method_cfg: dict, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg):
        super(DWMethod, self).__init__(name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg)

        method_cfg.pop('directory')  # self.directory
        method_cfg.pop('file_pattern')  # self.file_pattern
        self.annealing_times = method_cfg.pop('annealing_times')
        self.hyperparams = method_cfg
        self.num_hp = len(self.hyperparams)
        out_dir = Path(global_cfg['out_dir'])
        tts_out_file = out_dir / f'samp_{self.name}_tts_dat.pkl'
        if tts_out_file.is_file() and not global_cfg['overwrite']:
            logging.info(f"Loading sampler TTS data from {tts_out_file}.")
            with open(tts_out_file, 'rb') as f:
                hparam_array = pickle.load(f)
        else:
            # tts_shape = (len(self.rho_or_eps), len(self.llist), len(self.annealing_times), len(self.idxlist))
            hparam_names = list(self.hyperparams.keys())
            hparam_iters = list(self.hyperparams.values())
            hparam_lens = list(len(v) for v in self.hyperparams.values())
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
        self.dw_tts_array = hparam_array


class MCMethod(PGTMethodBase):
    def __init__(self, name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative, global_cfg):
        super(MCMethod, self).__init__(name, method_cfg, gs_energies, llist, idxlist, rho_or_eps, relative,
                                       global_cfg)

        out_dir = Path(global_cfg['out_dir'])
        tts_out_file = out_dir / f'{self.name}_tts_dat.pkl'

        self.tts_out_file = tts_out_file  # Save file for all TTS data
        self.tts_samp_file = out_dir / 'tts_samp.npz'  # save file for bootstrapped TTS samples
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

    def generate_bootstrap_scaling(self, rng: np.random.Generator):

        logging.info("Generating Bootstrap samples ... ")
        tts_samp_file = self.tts_samp_file
        pticm_tts_statistics: TTSStatistics = eval_boots_log_tts(self.tts_data, self.llist, n_boots=200, random_state=rng)
        logging.info(f"Saving to {tts_samp_file}")
        np.savez(tts_samp_file, L=np.asarray(self.llist), log_tts_mean=pticm_tts_statistics.mean,
                 log_tts_err=pticm_tts_statistics.err)

        return pticm_tts_statistics


class PGTScalingAnalysis:
    def __init__(self, config_file):
        with open(config_file) as f:
            cfg: dict = yaml.safe_load(f)
        self.cfg = cfg
        self.J = cfg['j']
        self.num_instances = cfg['num_instances']
        self.n_boots = cfg['n_boots']
        self.llist = cfg['llist']
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
            logging.info(f"Using relative epsilons:\n{self.rho_or_eps}")
        else:
            self.rho_or_eps = self.epsilon_arr
            logging.info(f"Using absolute epsilons:\n{self.rho_or_eps}")

        # Initialize RNG and bootstrap data
        self.rand_seed = cfg.get('random_seed', 0)
        if self.rand_seed is None or self.rand_seed == 0:
            logging.info("Random seed not set.")
            self.rand_seed = np.random.SeedSequence().entropy

        logging.info(f"Initialized random seed to {self.rand_seed}")
        self.rng = np.random.Generator(np.random.PCG64(self.rand_seed))
        self.mc_samplers = []
        self.dw_samplers = []

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
        self.mc_samplers = []
        for k, v in mc_samplers.items():
            self.mc_samplers.append(MCMethod(k, v, self.gs_energies, self.llist, self.idxlist, self.rho_or_eps,
                                             self.relative_epsilon, self.cfg))

    def load_dw_sampler_data(self):
        # Read the quantum samplers configurations
        dw_samplers: dict = self.cfg['quantum_samplers']
        self.dw_samplers = []
        for k, v in dw_samplers.items():
            self.dw_samplers.append(DWMethod(k, v, self.gs_energies, self.llist, self.idxlist, self.rho_or_eps,
                                             self.relative_epsilon, self.cfg))


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


if __name__ == "__main__":
    main()
