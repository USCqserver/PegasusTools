import numpy as np
import yaml
from . import tts


class TamcPtResults:
    def __init__(self, file, num_sweeps=None):
        # quick read
        try:
            with open(file) as f:
                lines = []
                for l in f:
                    if l.startswith("gs_states"):
                        break
                    lines.append(l)
                yaml_string = "\n".join(lines)
                pt_icm_info = yaml.safe_load(yaml_string)
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

    def cross_time(self, gs_energy, reps=0.0, eps=1.0e-4, maxsweeps=2 ** 31 - 1):
        lo_gs = np.logical_and((self.energies <= gs_energy + np.abs(gs_energy) * reps + eps), self.steps < maxsweeps)
        if not np.any(lo_gs):
            return np.inf, None
        cross_idx = np.argmax(lo_gs)
        cross_time = self.steps[cross_idx]
        return cross_time, cross_idx


def read_gs_energies(instance_template: str, l_list,  idx_list, ):
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


def optimal_tts_dist(tf, nts_arr, nsweeps):
    nsamps = len(nts_arr)
    # Optimal time-to solution by CDF
    ts_raw = tf * nts_arr / nsweeps
    pgs_arr = np.arange(1, nsamps + 1) / (nsamps + 1)
    tts_arr = tts(pgs_arr, ts_raw)
    opt_tts_idx = np.argmin(tts_arr)
    opt_tts = tts_arr[opt_tts_idx]
    opt_sweeps = nts_arr[opt_tts_idx]

    return opt_sweeps, opt_tts, opt_tts_idx


def read_nts_dist(file_fmt, num_reps, gs_energy, reps=0.0, eps=1.0e-4):
    """
    Evaluate optimal time-to-solution
    """
    nts_list = []
    time_list = []
    if not isinstance(reps, list):
        r_eps_list = [reps]
    else:
        r_eps_list = reps

    pt_res_list = []
    for r in range(num_reps):
        pt_res_list.append(TamcPtResults(file_fmt.format(r=r)))

    nsweeps = None
    reps_results = {'opt_sweeps': [], 'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for reps in r_eps_list:
        res_dict = {}
        cross_idx_list = []
        for r in range(num_reps):
            pt_res = pt_res_list[r]
            nts, cross_idx = pt_res.cross_time(gs_energy, eps=eps, reps=reps)
            cross_idx_list.append(cross_idx)
            t = pt_res.timing
            nts_list.append(nts)
            time_list.append(t)
            nsweeps = pt_res.num_sweeps

        avg_t = np.mean(time_list)
        nts_arr = np.asarray(nts_list)
        nts_arr = np.sort(nts_arr)[:num_reps - 1]

        opt_sweeps, opt_tts, opt_tts_idx = optimal_tts_dist(avg_t, nts_arr, nsweeps)

        reps_results['opt_sweeps'].append(opt_sweeps)
        reps_results['avg_t'].append(avg_t)
        reps_results['opt_tts'].append(opt_tts)
        reps_results['opt_tts_idx'].append(opt_tts_idx)
        reps_results['cross_idxs']. append(np.asarray(cross_idx_list))

    reps_results['cross_idxs'] = np.stack(reps_results['cross_idxs'])
    return reps_results


def import_pticm_dat(file_fmt, idxlist, gs_energies, reps=100, r_eps=0.0, eps=1.0e-4, maxsweeps=2 ** 31):
    pticm_dict = {'opt_sweeps': [], 'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for i, n in enumerate(idxlist):
        gs_e = gs_energies[i]
        reps_results = read_nts_dist(file_fmt.format(n=n), reps, gs_e,
                                     reps=r_eps, eps=eps)
        for k in pticm_dict.keys():
            pticm_dict[k].append(np.asarray(reps_results[k]))

    for k in pticm_dict.keys():
        pticm_dict[k] = np.stack(pticm_dict[k], axis=1)

    return pticm_dict

