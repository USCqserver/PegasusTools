import argparse
import numpy as np
import pegasustools as pgt
from pegasustools.util.adj import read_ising_adjacency
from pegasustools.util.sched import interpret_schedule
from pegasustools.pqubit import PegasusCellEmbedding
from dwave.system import DWaveSampler
import dimod
from dimod import SpinReversalTransformComposite

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--targets", nargs='+', type=int, help="The list of target logical states to keep track of.")
parser.add_argument("--tf", type=float, help="The anneal time for a simple annealing schedule.", default=20.0)
parser.add_argument("--schedule", nargs='+', help="Anneal schedule specification ")
parser.add_argument("--num-reads", type=int, default=64,
                    help="Number of solution readouts per repetition")
parser.add_argument("--reps", type=int, default=1,
                    help="Number of repetitions of data collection")
parser.add_argument("--rand-gauge", action='store_true',
                    help="Use a random gauge (spin reversal transformation) every repetition")
parser.add_argument("--rev-init", type=int,
                    help="Initial state for reverse annealing")
parser.add_argument("problem",
                    help="A cell problem, specified in a text file with three columns with the adjacency list")
parser.add_argument("output", help="Output file to print summary of samples.")

args = parser.parse_args()

# Import the problem and prepare the BQM
problem_file = args.problem
bqm = read_ising_adjacency(problem_file)
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
cell_sampler = PegasusCellEmbedding(16, dw_sampler, cache=True)
results_list = []

print("Sampling...")
for i in range(args.reps):
    print(f"{i+1}/{args.reps}")
    results = cell_sampler.sample(bqm, num_spin_reversal_transforms=1, num_reads=args.num_reads)
    results = results.aggregate()
    print(results)
    results_list.append(results.aggregate())

all_results: dimod.SampleSet = dimod.concatenate(results_list)
all_results = all_results.aggregate()
print(all_results)


