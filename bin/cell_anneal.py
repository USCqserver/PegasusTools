#!/usr/bin/env python3
import argparse

from pegasustools.app import add_general_arguments,add_cell_arguments, save_cell_results, run_sampler
from pegasustools.util.adj import read_ising_adjacency
from pegasustools.util.sched import interpret_schedule
from pegasustools.pqubit import PegasusCellEmbedding
from dwave.system import DWaveSampler
import dimod

parser = argparse.ArgumentParser()
add_general_arguments(parser)
add_cell_arguments(parser)

args = parser.parse_args()

# Import the problem and prepare the BQM
problem_file = args.problem
bqm = read_ising_adjacency(problem_file, args.scale_j)
print(bqm)

# Connect to the DW sampler
dw_sampler = DWaveSampler()

# Interpret and construct the annealing schedule
tf = args.tf
if args.schedule is not None:
    sched = interpret_schedule(args.tf, *args.schedule)
    print(sched)
    dw_sampler.validate_anneal_schedule(sched)
else:
    print(f"tf={args.tf}")
    sched = None
sched_kwags = {"anneal_schedule": sched} if sched is not None else {"annealing_time": args.tf}
dw_kwargs = {"num_spin_reversal_transforms": 1 if args.rand_gauge else 0,
             "num_reads": args.num_reads,
             "auto_scale": False}
print("Constructing cell embedding...")
if args.cell_p < 1.0:
    random_fill=args.cell_p
else:
    random_fill=None
cell_sampler = PegasusCellEmbedding(16, dw_sampler, random_fill=random_fill, cache=False)

aggr_results = run_sampler(cell_sampler, bqm, args, **dw_kwargs, **sched_kwags)
all_results: dimod.SampleSet = dimod.concatenate(aggr_results)
save_cell_results(all_results, sched, args)

