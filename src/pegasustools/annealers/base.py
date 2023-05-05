import abc
import argparse
import logging
import pathlib
import pickle

import dimod
from typing import Dict
import networkx as nx
import numpy as np
import pandas as pd
import yaml
import dwave_networkx as dnx
from itertools import combinations, product
from dimod.variables import Variables
from dwave.preprocessing import ScaleComposite
from dwave.system import DWaveSampler, AutoEmbeddingComposite, FixedEmbeddingComposite, LazyFixedEmbeddingComposite
from dwave.embedding import weighted_random
from pegasustools.util import concatenate
from pegasustools.util.sched import interpret_schedule
from pegasustools.util.adj import read_ising_adjacency, read_mapping
from pegasustools.embed import VariableMappingComposite
from pegasustools.embed.drawing import DrawEmbeddingWrapper
from pegasustools.embed.util import EmbeddingSummaryWrapper


class AnnealerModuleBase:

    def initialize_sampler(self, *args, **kwargs):
        """
        Initialize the sampler and the keywords to pass to the sampler.sample method
        """
        raise NotImplementedError

    def process_results(self, bqm, sampler, results):
        raise NotImplementedError

    @abc.abstractmethod
    def sampler_kwargs(self, d: Dict):
        raise NotImplementedError


class CompositeAnnealerModule(AnnealerModuleBase):
    def __init__(self, child_module: AnnealerModuleBase, **kwargs):
        self.child_module = child_module
        self._sampler_kwargs = {}
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False

    def initialize_sampler(self):
        return self.child_module.initialize_sampler()

    def process_results(self, bqm, sampler, results):
        return results

    def sampler_kwargs(self, d):
        """
        Recursively generate the kwargs to the sample method by updating the child kwargs
        :return:
        """
        self.child_module.sampler_kwargs(d)
        d.update(self._sampler_kwargs)


class DWaveAnnealerModule(AnnealerModuleBase):
    def __init__(self, **kwargs):
        self.tf = kwargs['tf']
        self.schedule = kwargs['schedule']
        self.num_reads = kwargs['num_reads']
        self.rand_gauge = kwargs['rand_gauge']
        self.auto_scale = kwargs['auto_scale']
        self._sampler_kwargs = {}

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        p = parser.add_argument_group("D-Wave Sampler Arguments")
        p.add_argument("--tf", type=float, help="The anneal time for a simple annealing schedule.", default=20.0)
        p.add_argument("--schedule", nargs='+', help="Anneal schedule specification ")
        p.add_argument("-n", "--num-reads", type=int, default=64,
                       help="Number of solution readouts per repetition")
        p.add_argument("-R", "--rand-gauge", action='store_true',
                       help="Use a random gauge (spin reversal transformation) every repetition")
        parser.add_argument("--auto-scale", action='store_true',
                            help="Auto-scale all couplings on the hardware graph (not compatible with --scale-j)"
                            )
        return p

    def initialize_sampler(self):
        sampler = DWaveSampler()
        dw_kwargs = self.initialize_schedule(sampler)
        dw_kwargs.update({
            "num_reads": self.num_reads,
            "auto_scale": self.auto_scale})
        if self.rand_gauge:
            dw_kwargs.update({"num_spin_reversal_transforms": 1})

        self._sampler_kwargs = dw_kwargs
        return sampler

    def process_results(self, bqm, sampler, results):
        return results

    def sampler_kwargs(self, d: Dict):
        d.update(self._sampler_kwargs)

    def initialize_schedule(self, sampler):
        tf = self.tf
        schedule = self.schedule
        # Interpret and construct the annealing schedule
        if schedule is not None:
            sched = interpret_schedule(tf, *schedule)
            print(sched)
            if 'anneal_schedule' in sampler.parameters:
                pass
                #sampler.validate_anneal_schedule(sched)
        else:
            print(f"tf={tf}")
            sched = None
        sched_kwargs = {"anneal_schedule": sched} if sched is not None else {"annealing_time": tf}
        return sched_kwargs


class AnnealerModuleRunner:
    def __init__(self, annealer_module: AnnealerModuleBase,
                 aggregate=True, preview_columns=None, **kwargs):
        self.annealer_module = annealer_module
        self.aggregate = aggregate
        self.preview_columns = preview_columns
        self.verbose = kwargs['verbose']
        self.reps = kwargs['reps']
        self.problem_file = kwargs['problem']
        self.output = kwargs['output']
        self.fmt = kwargs['format']
        self.qubo = kwargs['qubo']

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        p = parser.add_argument_group("Main Arguments")
        p.add_argument("--verbose", action='store_true')
        p.add_argument("--reps", type=int, default=1,
                       help="Number of repetitions of data collection")
        p.add_argument("--qubo", action='store_true')
        p.add_argument("--format", default="txt",
                       help="Input format for problem file. Defaults to 'txt' (whitespace delimited text)")
        p.add_argument("problem",
                       help="An annealing problem, specified in a text file with three columns with the adjacency list")
        p.add_argument("output", help="Prefix for output data")
        return p

    def initialize_bqm(self):
        sep = ',' if self.fmt == 'csv' else None
        bqm = read_ising_adjacency(self.problem_file, 1.0, sep, self.qubo)
        bqm = dimod.BQM(bqm)  # ensure dict-based BQM
        return bqm

    def main(self):
        """
        Main procedure of the Annealer Module
        :return: initialized BQM, initialized sampler, sampling results
        """
        bqm = self.initialize_bqm()
        sampler = self.annealer_module.initialize_sampler()
        sampler_kwargs = {}
        self.annealer_module.sampler_kwargs(sampler_kwargs)
        results = self.run_sampler(sampler, bqm, self.reps, aggregate=self.aggregate, **sampler_kwargs)
        results = self.annealer_module.process_results(bqm, sampler, results)
        self.save_results(self.output, results)
        return bqm, sampler, results

    @classmethod
    def run_sampler(cls, sampler, bqm, reps, aggregate=True, **sampler_kwargs):
        print("Sampling...")
        results_list = []
        # Collect the futures for each repetition
        for i in range(reps):
            results = sampler.sample(bqm, **sampler_kwargs)
            results_list.append(results)

        # Aggregate the results as they become available
        aggr_results = []
        for i, result in enumerate(results_list):
            print(f"{i + 1}/{reps}")
            samps = result.record.sample
            datvec: dict = result.data_vectors
            energies = datvec.pop('energy')
            num_occurrences = datvec.pop('num_occurrences')
            datvec['rep'] = np.full(energies.shape[0], i, dtype=int)
            res = dimod.SampleSet.from_samples((samps, result.variables), result.vartype, energies,
                                               info=result.info, num_occurrences=num_occurrences,
                                               aggregate_samples=False, sort_labels=False, **datvec)
            if aggregate:
                aggr_results.append(res.aggregate())
            else:
                aggr_results.append(res)
        # all_results = dimod.concatenate(aggr_results)
        return aggr_results

    def save_df(self, output, concat_results=None, aggr_results=None):
        if concat_results is None:
            concat_results = self.concat_results(aggr_results)

        num_vars = len(concat_results.variables)
        df = concat_results.to_pandas_dataframe()
        df_samples = df.iloc[:, :num_vars].astype("int8")
        df_properties = df.iloc[:, num_vars:]
        h5_file = output + ".h5"
        store = pd.HDFStore(h5_file, mode='w', complevel=5)
        store.append("samples", df_samples)
        store.append("info", df_properties)
        if 'dataframes' in concat_results.info:
            for k, v in concat_results.info['dataframes'].items():
                #store.put(f"pgt/{k}", v, format="table")
                store.append(f"pgt/{k}", v)
        store.close()
        return concat_results

    def preview_results(self, concat_results):
        # Preview results
        lo = concat_results.lowest()
        lo_df: pd.DataFrame = lo.to_pandas_dataframe()
        preview_cols = self.preview_columns
        if preview_cols is None:
            preview_cols = ['energy', 'rep', 'num_occurrences']
        # Preview only columns that exist in the df
        preview_cols = [c for c in preview_cols if c in lo_df.columns]
        print(lo_df.loc[:, preview_cols])
        num_gs = np.sum(lo.record.num_occurrences)
        total_reads = np.sum(concat_results.record.num_occurrences)
        print(f"The lowest energy appears in {num_gs}/{total_reads} samples")

    def save_results(self, output, aggr_results):
        concat_results = self.concat_results(aggr_results)
        self.preview_results(concat_results)
        self.save_df(output, concat_results, aggr_results)

    def concat_results(self, aggr_results):
        if 'dataframes' in aggr_results[0].info:
            concat_info = {}
            keys = aggr_results[0].info['dataframes'].keys()
            for k in keys:
                dfs = []
                for i, res in enumerate(aggr_results):
                    df = res.info['dataframes'][k]
                    #df["rep"] = np.full(len(df), i)
                    dfs.append(df)
                #concat_info[k] = pd.concat(dfs, ignore_index=True)
                concat_info[k] = pd.concat(
                    dfs, #ignore_index=True,
                    keys=list(range(len(aggr_results))),
                    names=["rep", "idx"])
            concat_results = concatenate(
                aggr_results, concat_info={'dataframes': concat_info})
        else:
            concat_results = concatenate(aggr_results)
        return concat_results


class ScaledModule(CompositeAnnealerModule):
    def __init__(self, child_module, scale_j=None, **kwargs):
        super(ScaledModule, self).__init__(child_module)
        self.scale_j = scale_j

    def initialize_sampler(self):
        sampler = self.child_module.initialize_sampler()

        scale_j = self.scale_j
        if scale_j is None:
            scale_j = 1.0

        sampler = ScaleComposite(sampler)
        scale_kwargs = {'scalar': 1.0 / scale_j}
        self._sampler_kwargs = scale_kwargs
        return sampler

    def sampler_kwargs(self, d):
        self.child_module.sampler_kwargs(d)
        if 'auto_scale' in d and self.scale_j is not None:
            if d['auto_scale']:
                print("Warning: --scale-j is being used with --auto-scale")
        d.update(self._sampler_kwargs)

    @classmethod
    def add_arguments(cls, parser):
        p = parser.add_argument_group("Scaling")
        p.add_argument("--scale-j", type=float, default=None,
                       help="Rescale all biases and couplings as J / scale_J")
        return p


class MinorEmbeddingModule(CompositeAnnealerModule):
    def __init__(self, child_module, **kwargs):
        super(MinorEmbeddingModule, self).__init__(child_module)
        self.embedding_name = kwargs['embedding_name']

        self.minor_embed = kwargs['minor_embed']
        self.chain_strength = kwargs['chain_strength']
        self.embedding_tries = kwargs['embedding_tries']
        self.embedding_threads = kwargs['embedding_threads']
        self.draw_embedding = kwargs['draw_embedding']
        self.track_chains = kwargs['track_chains']
        self.fixed_embedding = kwargs['fixed_embedding']

    @classmethod
    def add_arguments(cls, parser):
        p = parser.add_argument_group("Minor Embedding")
        p.add_argument("--minor-embed", action='store_true',
                       help="Minor-embed the instance to underlying graph")
        p.add_argument("--chain-strength", type=float, default=None,
                       help="Chain strength for minor-embed")
        p.add_argument("--embedding-tries", type=int, default=64)
        p.add_argument("--embedding-threads", type=int, default=1)
        p.add_argument("--draw-embedding", default=None)
        p.add_argument("--track-chains",  action='store_true')
        p.add_argument("--fixed-embedding", default=None)
        p.add_argument("--embedding-name", default='embedding')
        return p

    def initialize_sampler(self):
        sampler = self.child_module.initialize_sampler()
        if self.minor_embed:
            embedding_parameters = {
                'tries': self.embedding_tries, 'threads': self.embedding_threads
            }
            emb_kwargs = {
                'chain_strength': self.chain_strength,
                'chain_break_method': weighted_random,
                'return_embedding': True
            }
            save_embedding=None
            if self.fixed_embedding is not None:  # A fixed embedding file is specified
                emb_path = pathlib.Path(self.fixed_embedding)
                if emb_path.is_file():  # Load the embedding
                    logging.info(f"Loading embedding from {self.fixed_embedding}")
                    with open(emb_path, 'rb') as f:
                        emb = pickle.load(f)
                    if isinstance(emb, list):
                        emb = emb[0]
                    sampler = FixedEmbeddingComposite(sampler, embedding=emb)
                else:  # Save the first embedding found
                    logging.info(f"Embedding will be saved to {self.fixed_embedding}")
                    sampler = AutoEmbeddingComposite(sampler, embedding_parameters=embedding_parameters)
                    save_embedding = self.fixed_embedding
            else:
                sampler = AutoEmbeddingComposite(sampler, embedding_parameters=embedding_parameters)

            if self.track_chains or save_embedding is not None:
                sampler = EmbeddingSummaryWrapper(sampler, self.embedding_name, save_embedding=save_embedding)

            if self.draw_embedding is not None:
                sampler = DrawEmbeddingWrapper(sampler, self.draw_embedding, embedding_name=self.embedding_name)
        else:
            emb_kwargs = {}
        self._sampler_kwargs = emb_kwargs
        return sampler


class VariableMappingModule(CompositeAnnealerModule):
    def __init__(self, child_module, **kwargs):
        super(VariableMappingModule, self).__init__(child_module)
        self.mapping = kwargs['mapping']

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        p = parser.add_argument_group("Mapping")
        p.add_argument("--mapping", help="Apply a variable mapping to the problem before embedding.")

    def initialize_sampler(self):
        sub_sampler = self.child_module.initialize_sampler()
        if self.mapping is not None:
            variable_mapping = read_mapping(self.mapping)
            sampler = VariableMappingComposite(sub_sampler, variable_mapping=variable_mapping)
        else:
            sampler = sub_sampler
        return sampler
