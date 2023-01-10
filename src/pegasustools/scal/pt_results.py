import typing

import numpy as np
import yaml
import pickle
import pathlib as path
from typing import List, Dict
from . import tts
from .stats import reduce_mean, boots_percentile, TTSStatistics


class TamcPtResults:
    def __init__(self, *, file=None, from_dict=None, num_sweeps=None, quick_read=True):
        if file is None and from_dict is None:
            raise ValueError("One of 'file' or 'from_dict' must not be None")
        if from_dict is not None:
            pt_icm_info = from_dict
        else:
            try:
                file_path = path.Path(file)
                if file_path.suffix == ".pkl":
                    with open(file_path, 'rb') as f:
                        pt_icm_info = pickle.load(f)
                elif file_path.suffix == ".yml" or file_path.suffix == ".yaml":
                    with open(file) as f:
                        if quick_read:
                            lines = []
                            for l in f:
                                if l.startswith("gs_states"):
                                    break
                                lines.append(l)
                            yaml_string = "\n".join(lines)
                            pt_icm_info = yaml.safe_load(yaml_string)
                        else:
                            pt_icm_info = yaml.safe_load(f)
                else:
                    raise ValueError("Expect .pkl or .yml file")
            except Exception as e:
                print(f"Failed to parse {file}")
                raise e
                # timing in microseconds
        self.timing = pt_icm_info['timing']
        self.steps = np.asarray(pt_icm_info['gs_time_steps'])
        self.energies = np.asarray(pt_icm_info['gs_energies'])
        if 'params' in pt_icm_info.keys():
            self.num_sweeps = pt_icm_info['params']['num_sweeps']
        else:
            self.num_sweeps = num_sweeps
        
        if 'instance_size' in pt_icm_info.keys():
            self.instance_size = pt_icm_info['instance_size']
        else:
            self.instance_size = None

    def cross_time(self, gs_energy, epsilon=0.0, reps=0.0, tol=1.0e-4, maxsweeps=2 ** 31 - 1):
        lo_gs = np.logical_and((self.energies <= gs_energy + epsilon + np.abs(gs_energy) * reps + tol), self.steps < maxsweeps)
        if not np.any(lo_gs):
            return np.inf, None
        cross_idx = np.argmax(lo_gs)
        cross_time = self.steps[cross_idx]
        return cross_time, cross_idx


def read_gs_energies(instance_template: str, l_list,  idx_list ):
    gs_energies = np.zeros((len(l_list), len(idx_list)))
    for i, l in enumerate(l_list):
        for j, n in enumerate(idx_list):
            file = instance_template.format(l=l, n=n)
            #print(file)
            with open(file) as f:
                lines = []
                for line in f:
                    if line.startswith("gs_states"):
                        break
                    lines.append(line)
                yaml_string = "\n".join(lines)
                pt_icm_info = yaml.safe_load(yaml_string)
                energies = np.asarray(pt_icm_info['gs_energies'])
                gs_energies[i, j] = energies[-1]
    return gs_energies


def optimal_tts_dist(tf, nts_arr, nsweeps, ndiscard=1):
    """
    Evaluate the optimal time-to-solution
    Special case if len(nts_arr) == 1: simply return (nsweeps, tf, 0) for one-shoot TTS evaluation
    :param tf:
    :param nts_arr:
    :param nsweeps:
    :param ndiscard:
    :return:
    """
    nsamps = len(nts_arr)
    if nsamps > 1:
        # Optimal time-to solution by CDF estimate
        # where pgs_arr[0] =  1/(nsamps+1)... pgs_arr[nsamps-1] = (nsamps)/(nsamps+1)
        pgs_arr = np.arange(1, nsamps + 1) / (nsamps + 1)
        ts_raw = tf * nts_arr / nsweeps
        pgs_arr = pgs_arr[:nsamps-ndiscard]
        ts_raw = ts_raw[:nsamps-ndiscard]
        tts_arr = tts(pgs_arr, ts_raw)
        opt_tts_idx = np.argmin(tts_arr)
        opt_tts = tts_arr[opt_tts_idx]
        opt_sweeps = nts_arr[opt_tts_idx]

        return opt_sweeps, opt_tts, opt_tts_idx
    else:
        return nsweeps, tf, 0


def read_nts_dist(file_fmt, num_reps, gs_energy, reps=0.0, tol=1.0e-4):
    """
    Evaluate optimal time-to-solution
    """

    if not isinstance(reps, list):
        r_eps_list = [reps]
    else:
        r_eps_list = reps

    pt_res_list = []
    for r in range(num_reps):
        pt_res_list.append(TamcPtResults(file=file_fmt.format(r=r)))

    nsweeps = None
    reps_results = {'opt_sweeps': [], 'avg_t': [], 'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for reps in r_eps_list:
        nts_list = []
        time_list = []
        res_dict = {}
        cross_idx_list = []
        for r in range(num_reps):
            pt_res = pt_res_list[r]
            nts, cross_idx = pt_res.cross_time(gs_energy, reps=reps, tol=tol)
            cross_idx_list.append(cross_idx)
            t = pt_res.timing
            nts_list.append(nts)
            time_list.append(t)
            nsweeps = pt_res.num_sweeps

        time_arr = np.asarray(time_list)
        avg_t = np.mean(time_arr) / nsweeps
        nts_arr = np.asarray(nts_list)
        
        nts_arr = np.sort(nts_arr)[:num_reps - 1]
        time_arr = np.sort(time_arr)[:num_reps - 1]

        opt_sweeps, opt_tts, opt_tts_idx = optimal_tts_dist(time_arr, nts_arr, nsweeps)

        reps_results['opt_sweeps'].append(opt_sweeps)
        reps_results['avg_t'].append(avg_t)
        reps_results['opt_tts'].append(opt_tts)
        reps_results['opt_tts_idx'].append(opt_tts_idx)
        reps_results['cross_idxs']. append(np.asarray(cross_idx_list))

    reps_results['cross_idxs'] = np.stack(reps_results['cross_idxs'])
    return reps_results


def _read_nts_dist(pt_res_list: List[TamcPtResults], gs_energy, rhos_list, tol=1.0e-4,
                   maxsweeps=2**31-1, absolute=False):
    num_reps = len(pt_res_list)
    nsweeps = None
    reps_results = {'opt_sweeps': [], 'avg_t': [], 'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for rho in rhos_list:
        nts_list = []
        time_list = []
        cross_idx_list = []
        for r in range(num_reps):
            pt_res = pt_res_list[r]
            if absolute:
                eps = rho
            else:
                eps = rho * pt_res.instance_size
            nts, cross_idx = pt_res.cross_time(gs_energy, epsilon=eps, tol=tol, maxsweeps=maxsweeps)
            cross_idx_list.append(cross_idx)
            t = pt_res.timing
            nts_list.append(nts)
            time_list.append(t)
            nsweeps = pt_res.num_sweeps

        time_arr = np.asarray(time_list)
        avg_t = np.mean(time_arr) / nsweeps
        nts_arr = np.asarray(nts_list)

        nts_arr = np.sort(nts_arr)[:num_reps - 1]
        time_arr = np.sort(time_arr)[:num_reps - 1]

        opt_sweeps, opt_tts, opt_tts_idx = optimal_tts_dist(time_arr, nts_arr, nsweeps)

        reps_results['opt_sweeps'].append(opt_sweeps)
        reps_results['avg_t'].append(avg_t)
        reps_results['opt_tts'].append(opt_tts)
        reps_results['opt_tts_idx'].append(opt_tts_idx)
        reps_results['cross_idxs'].append(np.asarray(cross_idx_list))

    reps_results['cross_idxs'] = np.stack(reps_results['cross_idxs'])
    return reps_results


def read_nts_dist2(file_fmt, num_reps, gs_energy, rhos=0.0, tol=1.0e-4, absolute=False):
    """
    Evaluate optimal time-to-solution
    """

    if rhos is not None:
        if not isinstance(rhos, list):
            rhos_list = [rhos]
        else:
            rhos_list = rhos
    else:
        raise ValueError("rhos must not be None")

    pt_res_list = []
    for r in range(num_reps):
        pt_res_list.append(TamcPtResults(file=file_fmt.format(r=r), quick_read=False))

    return _read_nts_dist(pt_res_list, gs_energy, rhos_list, tol=tol, absolute=absolute)


def read_nts_dist_from_pkl(pkl_file, gs_energy, rhos=0.0, tol=1.0e-4, maxsweeps=2**31-1, absolute=False):
    """
    Evaluate optimal time-to-solution
    """

    if rhos is not None:
        if not isinstance(rhos, list):
            rhos_list = [rhos]
        else:
            rhos_list = rhos
    else:
        raise ValueError("rhos must not be None")

    pt_res_list = []
    with open(pkl_file, 'rb') as f:
        all_pkl = pickle.load(f)
    for r, d in enumerate(all_pkl):
        pt_res_list.append(TamcPtResults(from_dict=d, quick_read=False))

    return _read_nts_dist(pt_res_list, gs_energy, rhos_list, tol=tol, maxsweeps=maxsweeps, absolute=absolute)


def import_pticm_dat(file_fmt, idxlist, gs_energies, reps=100, r_eps=0.0, tol=1.0e-4, maxsweeps=2 ** 31):
    pticm_dict = {'opt_sweeps': [],'avg_t': [] ,'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for i, n in enumerate(idxlist):
        gs_e = gs_energies[i]
        reps_results = read_nts_dist(file_fmt.format(n=n), reps, gs_e,
                                     reps=r_eps, tol=tol)
        for k in pticm_dict.keys():
            pticm_dict[k].append(np.asarray(reps_results[k]))

    for k in pticm_dict.keys():
        pticm_dict[k] = np.stack(pticm_dict[k], axis=1)

    return pticm_dict


def import_pticm_dat2(file_fmt, idxlist, gs_energies, reps=100, rhos=0.0, tol=1.0e-4, maxsweeps=2 ** 31, absolute=False):
    pticm_dict = {'opt_sweeps': [],'avg_t': [] ,'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for i, n in enumerate(idxlist):
        gs_e = gs_energies[i]
        reps_results = read_nts_dist2(file_fmt.format(n=n), reps, gs_e,
                                      rhos=rhos, tol=tol, absolute=absolute)
        for k in pticm_dict.keys():
            pticm_dict[k].append(np.asarray(reps_results[k]))

    for k in pticm_dict.keys():
        pticm_dict[k] = np.stack(pticm_dict[k], axis=1)

    return pticm_dict


def import_pticm_dat_from_pkls(file_fmt, idxlist, gs_energies, epsilons=0.0, tol=1.0e-4,
                               maxsweeps=2 ** 31, absolute=False, fmt_kwargs=None):
    """

    :param file_fmt: pickle files with PT-ICM output data. Should be a format string with placeholder {n}, wher
    n will be subsituted with the indices from idxlist
    :param idxlist: list of indices of each instance
    :param gs_energies: ground state energy for each instance
    :param epsilons:  List of ground-state epsilons to consider
    :param tol: Tolerance to count an epsilon-crossing (energy <= gs_energy + epsilon
    :param maxsweeps: Limit the allowed number of sweeps to count an epsilon-crossing
    :param absolute: whether the ground-state epsilons are absolute (time-to-epsilon)
      or relative (time-to-rho, i.e. scale with the instance sizem)
    :return:
    """
    if fmt_kwargs is None:
        fmt_kwargs = {}
    pticm_dict = {'opt_sweeps': [],'avg_t': [] ,'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for i, n in enumerate(idxlist):
        gs_e = gs_energies[i]
        reps_results = read_nts_dist_from_pkl(file_fmt.format(n=n, **fmt_kwargs), gs_e,
                                      rhos=epsilons, tol=tol, maxsweeps=maxsweeps, absolute=absolute)
        for k in pticm_dict.keys():
            pticm_dict[k].append(np.asarray(reps_results[k]))

    for k in pticm_dict.keys():
        pticm_dict[k] = np.stack(pticm_dict[k], axis=1)

    return pticm_dict


class TimeToSolutionPT:
    def __init__(self, pkl_file_fmt, idxlist, gs_energies, epsilons=0.0, tol=1.0e-4, maxsweeps=2 ** 31, absolute=False):
        self.idxlist = idxlist
        self.gs_energies = gs_energies
        self.epsilons = epsilons
        self.tol = tol
        self.maxsweeps = maxsweeps
        self.absolute = absolute
        pticm_dict = import_pticm_dat_from_pkls(pkl_file_fmt, idxlist, gs_energies,  epsilons=epsilons, tol=tol,
                                                maxsweeps=maxsweeps, absolute=absolute)
        self.reps = len(pticm_dict['opt_tts'])
        self.opt_sweeps = pticm_dict['opt_sweeps']
        self.avg_t = pticm_dict['avg_t']
        self.opt_tts = pticm_dict['opt_tts']
        self.opt_tts_idx = pticm_dict['opt_tts_idx']
        self.cross_idx = pticm_dict['cross_idx']


def eval_boots_log_tts(pticm_tts: dict, l_list, n_boots=100, random_state=None):
    pticm_tts_log_med_lst = []
    pticm_tts_log_med_err_lst = []
    for l in l_list:
        opt_tts_arr = pticm_tts[l]['opt_tts'][:, np.newaxis, :]
        ttsl, ttslerr = reduce_mean(
            boots_percentile(np.log10(opt_tts_arr), 0.5,
                             n_boots=n_boots, random_state=random_state))
        pticm_tts_log_med_lst.append(ttsl)
        pticm_tts_log_med_err_lst.append(ttslerr)
    pticm_tts_log_med = np.stack(pticm_tts_log_med_lst)
    pticm_tts_log_med_err = np.stack(pticm_tts_log_med_err_lst)

    return TTSStatistics(pticm_tts_log_med, pticm_tts_log_med_err, None, l_list=l_list)


class TamcThermResults:
    def __init__(self):
        self.compression_level = None
        self.instance_size = None
        self.beta_arr = None
        self.samples = None
        self.e = None
        self.q = None
        self.suscept = None

    def avg_suscept(self, idxs):
        """
        Evaluates the average susceptibility of the components in idxs.
        The measurements suscept are the normalized (graph) Fourier transforms
        of the overlap
                K_j  = \sum_i w_i q_i
        The susceptibility returned is
                Chi = \sum_{j\in idxs} |K_j|^2
        If the weights are normalized by  \sum_i w_i^2 = N,
        then Chi / N corresponds to the usual definition of the
        spin-glass susceptibility Chi(k).
        :param idxs:
        :return:
        """
        nT = len(self.suscept)
        nsamps = len(self.suscept[0][0])
        chi_arr = np.zeros((nT, nsamps))
        for ti in range(nT):
            for i in idxs:
                chi_arr[ti, :] += self.suscept[ti][i]**2
        return chi_arr


def read_tamc_bin(file, version=1):
    import struct

    def deserialize_map(buffer, i, fn):
        n = struct.unpack('q', buffer[i: i + 8])[0]
        i += 8
        items = []
        for _ in range(n):
            i, x = fn(buffer, i)
            items.append(x)
        return i, items

    def deserialize_np_array(buffer, i, dt: np.dtype):
        n = struct.unpack('q', buffer[i: i+8])[0]
        dsize = dt.itemsize
        i += 8
        nbytes = n*dsize
        arr = np.frombuffer(buffer[i: i + n*dsize], dtype=dt)
        return i + n*dsize, arr

    deser_struct = [
        lambda buffer, i: (i+1, int(buffer[0])),  # compression_level u8
        lambda buffer, i: (i+8, struct.unpack('q', buffer[i:i+8])[0]),  # instance_size u64
        lambda buffer, i: deserialize_np_array(buffer, i, np.dtype('<f4')),  # beta_arr Vec<f32>
        lambda buffer, i: deserialize_map(
            buffer, i, lambda b2, i2: deserialize_map(
                b2, i2, lambda b3, i3: deserialize_np_array(b3, i3, np.dtype('u1')))
                                          ),       # samples: Vec<Vec<Vec<u8>>>
        lambda buffer, i: deserialize_map(
            buffer, i, lambda b2, i2: deserialize_np_array(b2, i2, np.dtype('<f4'))
        ),  # e: Vec<Vec<f32>>
        lambda buffer, i: deserialize_map(
            buffer, i, lambda b2, i2: deserialize_np_array(b2, i2, np.dtype('<i4'))
        ),  # q: Vec<Vec<i32>>,
    ]
    if version == 1:
        deser_struct += [
            lambda buffer, i: deserialize_map(
            buffer, i, lambda b2, i2: deserialize_map(
                b2, i2, lambda b3, i3: deserialize_np_array(b3, i3, np.dtype('f4')))
                                          ),       # samples: Vec<Vec<Vec<f32>>>
        ]

    results = TamcThermResults()
    with open(file, 'rb') as f:
        i = 0
        buf = f.read()
        i, results.compression_level = deser_struct[0](buf, i)
        i, results.instance_size = deser_struct[1](buf, i)
        i, results.beta_arr = deser_struct[2](buf, i)
        i, results.samples = deser_struct[3](buf, i)
        i, results.e = deser_struct[4](buf, i)
        i, results.q = deser_struct[5](buf, i)
        if version==1:
            i, results.suscept = deser_struct[6](buf, i)
    return results

