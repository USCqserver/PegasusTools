import numpy as np
import pandas as pd
import yaml


class DwRes:
    def __init__(self, file, inst=None, gs_energy=None):
        self.dw_res: pd.DataFrame = pd.read_hdf(file, "info")
        self.rep_groups = self.dw_res.groupby("rep")
        if inst is None:
            self.gs_energy = gs_energy
        else:
            with open(inst) as f:
                inst_info = yaml.safe_load(f)
            self.gs_energy = inst_info['gs_energy']

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

    def pgs_by_gauge(self, eps=1.0e-4, reps=0.0):
        if not isinstance(reps, list):
            return np.asarray(self._collect_pgs_by_gauge(eps, reps))
        else:
            pgs_by_reps = [np.asarray(self._collect_pgs_by_gauge(eps, re)) for re in reps]
            pgs_arr = np.stack(pgs_by_reps, axis=0)  # [reps, gauge, samps ]
            return pgs_arr

    def _collect_pgs_by_gauge(self, eps=1.0e-4, reps=0.0):
        rep_pgs = []
        for _, grp in self.rep_groups:
            m = np.sum(grp['num_occurrences'])
            mine_rows = grp[grp['energy'] <= self.gs_energy + np.abs(self.gs_energy) * reps + eps]
            n = np.sum(mine_rows['num_occurrences'])
            rep_pgs.append(n / m)
        return rep_pgs

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
        return cdf, ue
        #cdfs.append((counts, ue))
        #return cdfs

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
                try:
                    dw_res = DwRes(file_template.format(l=l, tf=tf, n=n), gs_energy=gs_energies[i, k])
                except FileNotFoundError as e:
                    print(e)
                    continue
                pgs_arr[:, i, j, k, :] = dw_res.pgs_by_gauge(reps=eps_r_list)
                if err_p:
                    errp_arr[i, j, k, :] = dw_res.error_p_by_gauge()

    if err_p:
        return pgs_arr, errp_arr
    else:
        return pgs_arr




