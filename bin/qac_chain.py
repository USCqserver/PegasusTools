import argparse

import dimod

from pegasustools.app import add_general_arguments, add_qac_arguments, run_sampler, save_cell_results
from pegasustools.qac import PegasusQACChainEmbedding
from pegasustools.util.adj import read_ising_adjacency
from pegasustools.util.sched import interpret_schedule
from dwave.system import DWaveSampler

parser = argparse.ArgumentParser()
add_general_arguments(parser)
add_qac_arguments(parser)
args = parser.parse_args()

problem_file = args.problem
tf = args.tf
bqm = read_ising_adjacency(problem_file, args.scale_j)
print(bqm)

dw_sampler = DWaveSampler()

# Interpret and construct the annealing schedule
if args.schedule is not None:
    sched = interpret_schedule(args.tf, *args.schedule)
    print(sched)
    dw_sampler.validate_anneal_schedule(sched)
else:
    print(f"tf={args.tf}")
    sched = None
if args.verbose:
    print(f"QAC Penalty: {args.qac_penalty}")
    print(f"QAC Problem scale: {args.qac_scale}")
qac_args = {
    "qac_penalty_strength": args.qac_penalty,
    "qac_problem_scale": args.qac_scale,
}
sched_kwags = {"anneal_schedule": sched} if sched is not None else {"annealing_time": args.tf}
dw_kwargs = {"num_spin_reversal_transforms": 1 if args.rand_gauge else 0,
             "num_reads": args.num_reads,
             "auto_scale": False}

print("Constructing QAC chain...")
qac_sampler = PegasusQACChainEmbedding(16, dw_sampler)

aggr_results = run_sampler(qac_sampler, bqm, args, **qac_args, **dw_kwargs, **sched_kwags)
all_results = dimod.concatenate(aggr_results)
save_cell_results(all_results, sched, args)
