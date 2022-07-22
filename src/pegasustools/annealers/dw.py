import argparse
import dimod
import numpy as np
import pandas as pd
from dwave.preprocessing import ScaleComposite
from dwave.system import DWaveSampler, AutoEmbeddingComposite
from pegasustools.util.sched import interpret_schedule
from pegasustools.util.adj import read_ising_adjacency


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
        bqm = self.initialize_bqm()
        sampler = self.initialize_sampler()
        kwargs = self.generate_kwargs_dict()
        results = self.run_sampler(sampler, bqm, self.args.reps, aggregate=self.aggregate, **kwargs)
        self.save_results(self.args.output, results)
        return bqm, sampler, results

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
        all_results = dimod.concatenate(aggr_results)
        return all_results

    @classmethod
    def save_results(cls, output, all_results, mapping=None, preview_cols=None):
        if mapping is not None:
            all_results.relabel_variables(mapping)
        # Preview results
        lo = all_results.lowest()
        lo_df: pd.DataFrame = lo.to_pandas_dataframe()
        if preview_cols is None:
            preview_cols = ['energy', 'rep', 'num_occurrences']

        print(lo_df.loc[:, preview_cols])
        num_gs = np.sum(lo.record.num_occurrences)
        total_reads = np.sum(all_results.record.num_occurrences)
        print(f"The lowest energy appears in {num_gs}/{total_reads} samples")
        # samps_df = df = pd.DataFrame(all_results.record.sample, columns=all_results.variables)
        num_vars = len(all_results.variables)
        # Save results
        df = all_results.to_pandas_dataframe()
        df_samples = df.iloc[:, :num_vars].astype("int8")
        df_properties = df.iloc[:, num_vars:]
        h5_file = output + ".h5"
        store = pd.HDFStore(h5_file, mode='w', complevel=5)
        store.append("samples", df_samples)
        store.append("info", df_properties)
        store.close()


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
                     "auto_scale": False}
        self.kwargs_list.append(dw_kwargs)
        return sampler

    def initialize_schedule(self, sampler):
        args = self.args
        # Interpret and construct the annealing schedule
        if args.schedule is not None:
            sched = interpret_schedule(args.tf, *args.schedule)
            print(sched)
            if 'anneal_schedule' in sampler.parameters:
                sampler.validate_anneal_schedule(sched)
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
        p.add_argument("problem",
                       help="An annealing problem, specified in a text file with three columns with the adjacency list")
        p.add_argument("output", help="Prefix for output data")


class ScaledModule(AnnealerModule):
    def initialize_sampler(self):
        sampler = super(ScaledModule, self).initialize_sampler()
        args = self.args
        sampler = ScaleComposite(sampler)
        scale_kwargs = {'scalar': 1.0 / args.scale_j}
        self.kwargs_list.append(scale_kwargs)
        return sampler

    @classmethod
    def add_arguments(cls, parser):
        super(ScaledModule, cls).add_arguments(parser)
        parser.add_argument("--scale-j", type=float, default=1.0,
                       help="Rescale all biases and couplings as J / scale_J")


class MinorEmbeddingModule(AnnealerModule):
    def initialize_sampler(self):
        sampler = super(MinorEmbeddingModule, self).initialize_sampler()
        args = self.args
        if args.minor_embed:
            sampler = AutoEmbeddingComposite(sampler)
            emb_kwargs = {
                'chain_strength': args.chain_strength
            }
        else:
            emb_kwargs = {}
        self.kwargs_list.append(emb_kwargs)
        return sampler

    @classmethod
    def add_arguments(cls, parser):
        super(MinorEmbeddingModule, cls).add_arguments(parser)
        p = parser.add_argument_group("Minor Embedding")
        p.add_argument("--minor-embed", action='store_true',
                       help="Minor-embed the instance to the QAC graph")
        p.add_argument("--chain-strength", type=float, default=None,
                       help="Chain strength for minor-embed")
        p.add_argument("--save-embedding", default=None)