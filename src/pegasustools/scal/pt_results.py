import typing

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
    reps_results = {'opt_sweeps': [], 'avg_t': [], 'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
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


def import_pticm_dat(file_fmt, idxlist, gs_energies, reps=100, r_eps=0.0, eps=1.0e-4, maxsweeps=2 ** 31):
    pticm_dict = {'opt_sweeps': [],'avg_t': [] ,'opt_tts': [], 'opt_tts_idx': [], 'cross_idxs': []}
    for i, n in enumerate(idxlist):
        gs_e = gs_energies[i]
        reps_results = read_nts_dist(file_fmt.format(n=n), reps, gs_e,
                                     reps=r_eps, eps=eps)
        for k in pticm_dict.keys():
            pticm_dict[k].append(np.asarray(reps_results[k]))

    for k in pticm_dict.keys():
        pticm_dict[k] = np.stack(pticm_dict[k], axis=1)

    return pticm_dict


class TamcThermResults:
    def __init__(self):
        self.compression_level = None
        self.instance_size = None
        self.beta_arr = None
        self.samples = None
        self.e = None
        self.q = None
        self.suscept = None


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

