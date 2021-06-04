import argparse
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rcf
import pegasustools as pgt
from pegasustools.util.adj import read_ising_adjacency
from pegasustools.util.sched import interpret_schedule
from pegasustools.pqubit import PegasusCellEmbedding
from dwave.system import DWaveSampler
import dimod

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument("--targets", nargs='+', type=int, help="The list of target logical states to keep track of.")
parser.add_argument("--tf", type=float, help="The anneal time for a simple annealing schedule.", default=20.0)
parser.add_argument("--schedule", nargs='+', help="Anneal schedule specification ")
parser.add_argument("-n","--num-reads", type=int, default=64,
                    help="Number of solution readouts per repetition")
parser.add_argument("--reps", type=int, default=1,
                    help="Number of repetitions of data collection")
parser.add_argument("-R", "--rand-gauge", action='store_true',
                    help="Use a random gauge (spin reversal transformation) every repetition")
parser.add_argument("--rev-init", type=int,
                    help="Initial state for reverse annealing")
parser.add_argument("--scale-j", type=float, default=1.0,
                    help="Rescale all biases and couplings as J / scale_J")
parser.add_argument("problem",
                    help="A cell problem, specified in a text file with three columns with the adjacency list")
parser.add_argument("output", help="Prefix for output data")

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
    sched = None
print("Constructing cell embedding...")
cell_sampler = PegasusCellEmbedding(16, dw_sampler, cache=False)


def profile():
    print("Profiling cell embedding...")
    import pstats, cProfile
    cProfile.runctx("PegasusCellEmbedding(16, dw_sampler, cache=False)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumulative").print_stats()

    print("Profiling sampling...")
    cProfile.runctx("cell_sampler.sample(bqm, num_spin_reversal_transforms=1, num_reads=args.num_reads).aggregate()",
                    globals(), locals(), "samp.prof")
    s = pstats.Stats("samp.prof")
    s.strip_dirs().sort_stats("cumulative").print_stats()

#profile()

results_list = []

print("Sampling...")
# Collect the futures for each repetition
for i in range(args.reps):
    results = cell_sampler.sample(bqm, num_spin_reversal_transforms=1, num_reads=args.num_reads)
    results_list.append(results)

# Aggregate the results as they become available
aggr_results = []
for i, result in enumerate(results_list):
    print(f"{i+1}/{args.reps}")
    aggr_results.append(result.aggregate())

all_results: dimod.SampleSet = dimod.concatenate(aggr_results)
all_results = all_results.aggregate()
sample = all_results.record.sample
n_arr = pgt.util.ising_to_intlabel(sample)
all_results = dimod.append_data_vectors(all_results, blabel=n_arr)
print(all_results)

df : pd.DataFrame = all_results.to_pandas_dataframe()
df2 = df[["blabel", "num_occurrences", "energy"]]
print(df[["blabel", "num_occurrences", "energy"]])
csv_path = f"{args.output}_samps.csv"
sched_path = f"{args.output}_sched.csv"
with open(csv_path, 'w') as f:
    df2.to_csv(f, index=False)

sched_arr = np.asarray(sched)
np.savetxt(sched_path, sched_arr)
#print(sample)
#print(n_arr)


