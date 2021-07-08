import argparse
import networkx as nx
import dimod
from dimod.bqm import AdjVectorBQM
import numpy as np
from pegasustools.app import add_general_arguments, add_qac_arguments, run_sampler, save_cell_results
from pegasustools.qac import PegasusQACEmbedding
from pegasustools.util.adj import read_ising_adjacency
from pegasustools.util.sched import interpret_schedule
from dwave.system import DWaveSampler

parser = argparse.ArgumentParser(
    description="Generate random instances for the native QAC topology"
)
#add_general_arguments(parser)
#add_qac_arguments(parser)
parser.add_argument("dest")
args = parser.parse_args()


dw_sampler = DWaveSampler()
qac_sampler = PegasusQACEmbedding(16, dw_sampler)
g = nx.Graph()

lin = {n: 0.0 for n in qac_sampler.nodelist}
for n in qac_sampler.nodelist:
    g.add_node(n, bias=0.0)
qua = {}
num_edges = len(qac_sampler.edgelist)
# Generate random couplings
rand_js = np.random.randint(1, 4, [num_edges])*(2*np.random.randint(0, 2, [num_edges])-1)
rand_js = rand_js / 6.0
for e, j in zip(qac_sampler.edgelist, rand_js):
    qua[e] = j
    g.add_edge(e[0], e[1], coupling=j)

bqm = AdjVectorBQM(lin, qua, dimod.SPIN)

tflist = [1.0, 5.0, 25.0]
samples_qac = [
    qac_sampler.sample(bqm, encoding='qac', qac_penalty_strength=0.3, qac_problem_scale=1.0, num_reads=20,
                       auto_scale=False, annealing_time=tf)
    for tf in tflist
]
samples_c = [
    qac_sampler.sample(bqm, encoding='all', num_reads=20, auto_scale=False, annealing_time=tf)
    for tf in tflist
]
print("QAC")
for tf, s in zip(tflist, samples_qac):
    print(f"tf = {tf}")
    print(s.truncate(10))
print("ALL")
for tf, s in zip(tflist, samples_c):
    print(f"tf = {tf}")
    print(s.truncate(10))
