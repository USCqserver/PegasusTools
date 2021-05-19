import argparse
import numpy as np
import pegasustools as pgt
from pegasustools.util.adj import read_ising_adjacency
from pegasustools.util.sched import interpret_schedule

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--targets", nargs='+', type=int, help="The list of target logical states to keep track of.")
parser.add_argument("--tf", type=float, help="The anneal time for a simple annealing schedule.", default=20.0)
parser.add_argument("--schedule", nargs='+', help="Anneal schedule specification ")
parser.add_argument("--rev-init")
parser.add_argument("problem",
                    help="A cell problem, specified in a text file with three columns with the adjacency list")
parser.add_argument("output", help="Output file to print summary of samples.")

args = parser.parse_args()

# Import the problem and prepare the BQM
problem_file = args.problem
bqm = read_ising_adjacency(problem_file)
print(bqm)

# Interpret and construct the annealing schedule
tf = args.tf
if args.schedule is not None:
    sched = interpret_schedule(args.tf, *args.schedule)
else:
    sched = None



