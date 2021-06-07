import argparse
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rcf
import pegasustools as pgt
from pegasustools.app import add_general_argument, save_cell_results
from pegasustools.util.adj import read_ising_adjacency
from pegasustools.util.sched import interpret_schedule
from pegasustools.pqubit import PegasusCellEmbedding
from dwave.system import DWaveSampler
import dimod

parser = argparse.ArgumentParser()
add_general_argument(parser)

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
print("Constructing cell embedding...")
cell_sampler = PegasusCellEmbedding(16, dw_sampler, cache=False)

results_list = []

print("Sampling...")
# Collect the futures for each repetition
for i in range(args.reps):
    sched_kwags = {"anneal_schedule": sched} if sched is not None else {"annealing_time": args.tf}
    results = cell_sampler.sample(bqm, num_spin_reversal_transforms=1, num_reads=args.num_reads,
                                  auto_scale=False, **sched_kwags)
    results_list.append(results)

# Aggregate the results as they become available
aggr_results = []
for i, result in enumerate(results_list):
    print(f"{i+1}/{args.reps}")
    aggr_results.append(result.aggregate())

all_results: dimod.SampleSet = dimod.concatenate(aggr_results)
save_cell_results(all_results, sched, args)
#print(sample)
#print(n_arr)


