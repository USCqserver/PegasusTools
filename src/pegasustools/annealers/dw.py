import argparse
import dimod
import networkx as nx
import numpy as np
import pandas as pd
import yaml
import dwave_networkx as dnx
from itertools import combinations, product
from dimod.variables import Variables
from dwave.preprocessing import ScaleComposite
from dwave.system import DWaveSampler, AutoEmbeddingComposite
from dwave.embedding import weighted_random
from pegasustools.util.sched import interpret_schedule
from pegasustools.util.adj import read_ising_adjacency, read_mapping
from pegasustools.embed import VariableMappingComposite
from pegasustools.embed.drawing import draw_minor_embedding


class AnnealerModuleBase:
    def __init__(self, args, aggregate=True):
        self.args = args
        self.aggregate = aggregate
        self.kwargs_list = []

    def generate_kwargs_dict(self):
        kwargs = {}
        for d in reversed(self.kwargs_list):
            kwargs.update(d)
        return kwargs

    def main(self):
        """
        Main procedure of the Annealer Module
        :return: initialized BQM, initialized sampler, sampling results
        """
        bqm = self.initialize_bqm()
        sampler = self.initialize_sampler()
        kwargs = self.generate_kwargs_dict()
        results = self.run_sampler(sampler, bqm, self.args.reps, aggregate=self.aggregate, **kwargs)
        results = self.process_results(bqm, sampler, results)
        self.save_results(self.args.output, results)
        return bqm, sampler, results

    def process_results(self, bqm, sampler, results):
        """
        Any additional post-processing before saving the results
        :param bqm:
        :param sampler:
        :param results:
        :return:
        """
        return results

    def initialize_bqm(self):
        raise NotImplementedError

    def initialize_sampler(self):
        raise NotImplementedError

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
            datvec['rep'] = np.full(energies.shape[0], i, dtype=np.int)
            res = dimod.SampleSet.from_samples((samps, result.variables), result.vartype, energies,
                                               info=result.info, num_occurrences=num_occurrences,
                                               aggregate_samples=False, sort_labels=False, **datvec)
            if aggregate:
                aggr_results.append(res.aggregate())
            else:
                aggr_results.append(res)
        #all_results = dimod.concatenate(aggr_results)
        return aggr_results

    def save_df(self, output, concat_results=None, aggr_results=None):
        if concat_results is None:
            concat_results = dimod.concatenate(aggr_results)
        num_vars = len(concat_results.variables)
        df = concat_results.to_pandas_dataframe()
        df_samples = df.iloc[:, :num_vars].astype("int8")
        df_properties = df.iloc[:, num_vars:]
        h5_file = output + ".h5"
        store = pd.HDFStore(h5_file, mode='w', complevel=5)
        store.append("samples", df_samples)
        store.append("info", df_properties)
        store.close()
        return concat_results

    def preview_results(self, concat_results, preview_cols=None):
        # Preview results
        lo = concat_results.lowest()
        lo_df: pd.DataFrame = lo.to_pandas_dataframe()
        if preview_cols is None:
            preview_cols = ['energy', 'rep', 'num_occurrences']

        print(lo_df.loc[:, preview_cols])
        num_gs = np.sum(lo.record.num_occurrences)
        total_reads = np.sum(concat_results.record.num_occurrences)
        print(f"The lowest energy appears in {num_gs}/{total_reads} samples")

    def save_results(self, output, aggr_results, mapping=None, preview_cols=None):
        concat_results = dimod.concatenate(aggr_results)
        if mapping is not None:
            concat_results.relabel_variables(mapping)
            for r in aggr_results:
                r.relabel_variables(mapping)
        self.preview_results(concat_results, preview_cols)
        self.save_df(output, concat_results, aggr_results)


class AnnealerModule(AnnealerModuleBase):
    def initialize_bqm(self):
        args = self.args
        problem_file = args.problem
        sep = ',' if args.format == 'csv' else None
        bqm = read_ising_adjacency(problem_file, 1.0, sep, args.qubo)
        bqm = dimod.BQM(bqm)  # ensure dict-based BQM
        return bqm

    def initialize_sampler(self):
        args = self.args
        sampler = DWaveSampler()
        self.initialize_schedule(sampler)
        dw_kwargs = {"num_spin_reversal_transforms": 1 if args.rand_gauge else 0,
                     "num_reads": args.num_reads,
                     "auto_scale": args.auto_scale}

        self.kwargs_list.append(dw_kwargs)
        return sampler

    def initialize_schedule(self, sampler):
        args = self.args
        # Interpret and construct the annealing schedule
        if args.schedule is not None:
            sched = interpret_schedule(args.tf, *args.schedule)
            print(sched)
            if 'anneal_schedule' in sampler.parameters:
                pass
                #sampler.validate_anneal_schedule(sched)
        else:
            print(f"tf={args.tf}")
            sched = None
        sched_kwargs = {"anneal_schedule": sched} if sched is not None else {"annealing_time": args.tf}
        self.kwargs_list.append(sched_kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        p = parser.add_argument_group("Annealer")
        p.add_argument("-v", "--verbose", action='store_true')
        p.add_argument("--tf", type=float, help="The anneal time for a simple annealing schedule.", default=20.0)
        p.add_argument("--schedule", nargs='+', help="Anneal schedule specification ")
        p.add_argument("-n", "--num-reads", type=int, default=64,
                       help="Number of solution readouts per repetition")
        p.add_argument("--reps", type=int, default=1,
                       help="Number of repetitions of data collection")
        p.add_argument("-R", "--rand-gauge", action='store_true',
                       help="Use a random gauge (spin reversal transformation) every repetition")
        p.add_argument("--qubo", action='store_true')
        p.add_argument("--format", default="txt",
                            help="Input format for problem file. Defaults to 'txt' (whitespace delimited text)")
        parser.add_argument("--auto-scale", action='store_true',
                            help="Auto-scale all couplings on the hardware graph (not compatible with --scale-j)"
                            )
        p.add_argument("problem",
                       help="An annealing problem, specified in a text file with three columns with the adjacency list")
        p.add_argument("output", help="Prefix for output data")
        return p


class ScaledModule(AnnealerModule):
    def initialize_sampler(self):
        sampler = super(ScaledModule, self).initialize_sampler()
        args = self.args
        if args.auto_scale and args.scale_j is not None:
            print("Warning: --scale-j is being used with --auto-scale")

        if args.scale_j is None:
            scale_j = 1.0
        else:
            scale_j = args.scale_j

        sampler = ScaleComposite(sampler)
        scale_kwargs = {'scalar': 1.0 / scale_j}
        self.kwargs_list.append(scale_kwargs)
        return sampler

    @classmethod
    def add_arguments(cls, parser):
        p = super(ScaledModule, cls).add_arguments(parser)
        p.add_argument("--scale-j", type=float, default=None,
                       help="Rescale all biases and couplings as J / scale_J")
        return p


class MinorEmbeddingModule(AnnealerModule):
    def __init__(self, *args, **kwargs):
        super(MinorEmbeddingModule, self).__init__(*args, **kwargs)
        self.child_graph = None
        self.logical_graph = None

    def main(self):
        bqm, sampler, results = super(MinorEmbeddingModule, self).main()
        if self.args.draw_embedding:
            pgraph = dnx.pegasus_graph(16)
            for i in range(len(results)):
                draw_minor_embedding(self.args.output+f'_{i}_embedding.pdf', pgraph, results[i], bqm,
                                          results[i].info['embedding_context']['embedding'])
        return bqm, sampler, results

    def initialize_sampler(self):
        sampler = super(MinorEmbeddingModule, self).initialize_sampler()
        args = self.args
        if args.minor_embed:
            embedding_parameters = {
                'tries': args.embedding_tries, 'threads': args.embedding_threads
            }
            emb_kwargs = {
                'chain_strength': args.chain_strength,
                'chain_break_method': weighted_random,
                'return_embedding': True
            }
            if args.initial_chains is not None:
                with open(args.initial_chains) as f:
                    initial_chains = yaml.safe_load(f)
                embedding_parameters['initial_chains'] = initial_chains
            sampler = AutoEmbeddingComposite(sampler, embedding_parameters=embedding_parameters)
        else:
            emb_kwargs = {}
        self.kwargs_list.append(emb_kwargs)
        return sampler

    def save_df(self, output, concat_results=None, aggr_results=None):
        super(MinorEmbeddingModule, self).save_df(output, concat_results, aggr_results)
        h5_file = output + ".h5"
        if self.args.minor_embed:
            store = pd.HDFStore(h5_file, mode='a')
            emb_dat = {"avg_chain_length": [], "max_chain_length": []}
            for i in range(len(aggr_results)):
                emb = aggr_results[i].info['embedding_context']['embedding']
                chain_lens = [len(ci) for ci in emb.values()]
                emb_dat['avg_chain_length'].append(np.mean(chain_lens))
                emb_dat['max_chain_length'].append(np.max(chain_lens))
            emb_df = pd.DataFrame(emb_dat)
            print(emb_df.describe())
            store.append("embedding_info", emb_df)
            store.close()

    @classmethod
    def add_arguments(cls, parser):
        super(MinorEmbeddingModule, cls).add_arguments(parser)
        p = parser.add_argument_group("Minor Embedding")
        p.add_argument("--minor-embed", action='store_true',
                       help="Minor-embed the instance to underlying graph")
        p.add_argument("--chain-strength", type=float, default=None,
                       help="Chain strength for minor-embed")
        p.add_argument("--embedding-tries", type=int, default=64)
        p.add_argument("--embedding-threads", type=int, default=1)
        p.add_argument("--draw-embedding", action='store_true')
        return p


class VariableMappingModule(AnnealerModule):
    def __init__(self, *args, **kwargs):
        super(VariableMappingModule, self).__init__(*args, **kwargs)
        self.mapping_l2n = None
        self.mapping_n2l = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        p = super(VariableMappingModule, cls).add_arguments(parser)
        p.add_argument("--mapping", help="Apply a variable mapping to the problem before embedding.")

    def initialize_sampler(self):
        sub_sampler = super(VariableMappingModule, self).initialize_sampler()
        if self.args.mapping is not None:
            variable_mapping = read_mapping(self.args.mapping)
            sampler = VariableMappingComposite(sub_sampler, variable_mapping=variable_mapping)
        else:
            sampler = sub_sampler
        return sampler
