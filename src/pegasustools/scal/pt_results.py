import numpy as np
import yaml


class TamcPtResults:
    def __init__(self, file, num_sweeps=None):
        # quick read
        with open(file) as f:
            lines = []
            for l in f:
                if l.startswith("gs_states"):
                    break
                lines.append(l)
            yaml_string = "\n".join(lines)
            pt_icm_info = yaml.safe_load(yaml_string)
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
    gs_energies = np.zeros(len(l_list), len(idx_list))
    for i, l in enumerate(l_list):
        for j, n in enumerate(idx_list):
            file = instance_template.format(l=l, n=n)
            with open(file) as f:
                lines = []
                for l in f:
                    if l.startswith("gs_states"):
                        break
                    lines.append(l)
                yaml_string = "\n".join(lines)
                pt_icm_info = yaml.safe_load(yaml_string)
                energies = np.asarray(pt_icm_info['gs_energies'])
                gs_energies[i, j] = energies[-1]
    return gs_energies