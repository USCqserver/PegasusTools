import argparse
import numpy as np
import pegasustools as pgt
from pegasustools.util.adj import read_ising_adjacency

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--targets", nargs='+', type=int, help="The list of target logical states to keep track of.")
parser.add_argument("--tf", type=float, help="The anneal time for a simple annealing schedule.")
parser.add_argument("--schedule", nargs='+', help="Anneal schedule specification ")
parser.add_argument("problem", required=True,
                    help="A cell problem, specified in a text file with three columns with the adjacency list")
parser.add_argument("output", required=True, help="Output file to print summary of samples.")

args = parser.parse_args()

# Import the problem and prepare the BQM
problem_file = args.problem
bqm = read_ising_adjacency(problem_file)





