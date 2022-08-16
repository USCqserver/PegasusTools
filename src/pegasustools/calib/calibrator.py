import networkx as nx
import numpy as np
from dimod import Composite, Sampler, ComposedSampler, BinaryQuadraticModel, SampleSet
from dwave.system import DWaveSampler


def sampler_calibrator(sampler: DWaveSampler, g: nx.Graph, num_iterations=256,
                       samples_per_iterations=100,
                       bias_step_size=5e-6,
                       coupling_step_size=0.2,
                       coupling_momentum=0.9,
                       **parameters):
    nodes = sampler.nodelist
    edges = sampler.edgelist
    g = sampler.to_networkx_graph()
    h = {n: 0.0 for n in nodes}
    J = {e: -1.0 for e in edges}
    bqm = BinaryQuadraticModel.from_ising(h, J)

    eta_coupl = coupling_step_size
    g_coupl = coupling_momentum
    fb = np.zeros(sampler.properties['num_qubits'])
    fb_grad = np.zeros(sampler.properties['num_qubits'])
    for i in range(num_iterations):
        samps: SampleSet = sampler.sample(
            bqm, flux_drift_compensation=False, flux_biases=list(fb),
            num_reads=samples_per_iterations)
        s = samps.record.sample
        avg_m = np.mean(s, axis=0)
        cor = {(i, j): np.mean(s[:, i]*s[: j]) for (i, j) in J.keys()}
        frust = { k: (-v+1.0)/2.0 for k,v in cor.items()}
        # update the flux biases
        fb_grad = g_coupl * fb_grad + (1.0 - g_coupl) * eta_coupl * avg_m
        fb = fb - fb_grad
        # update the coupling biases

class CalibratorBase:
    """
    Base class for generalized calibration on a sampler
    """
    def __init__(self, sampler: Sampler, biases, step_sizes, momenta, num_iterations=256):
        self.sampler = sampler
        self.biases=biases
        self.step_sizes=step_sizes
        self.momenta=momenta
        self.num_iterations = 256

    def initialize_biases(self):
        """
        Generate the initial biases and gradients (which should be all-zeros) as dictionaries
        :return: init_biases, init_gradients
        """
        raise NotImplementedError

    def sample_correlations(self, bqm: BinaryQuadraticModel, **sampler_kwargs):
        J = bqm.quadratic
        samps: SampleSet = self.sampler.sample(bqm, **sampler_kwargs)
        s = samps.record.sample
        avg_m = np.mean(s, axis=0)
        cor = {(i, j): np.mean(s[:, i] * s[: j]) for (i, j) in J.keys()}
        return avg_m, cor


class DWaveFMCalibrator(CalibratorBase):
    """
    Calibrate a DWave sampler at the hardware level using a ferromagnetic instance
    """
    def __init__(self, sampler: DWaveSampler,
                 calib_qubits,
                 bias_step_size=5e-6,
                 offset_step_size=0.2,
                 coupling_momentum=0.9,
                 ):
        biases = ["flux_biases", "anneal_offset"]
        step_sizes = {"flux_biases": bias_step_size, "anneal_offset": offset_step_size}
        momenta = {"flux_biases": 0.9, "anneal_offset": 0.9}
        super(DWaveFMCalibrator, self).__init__(sampler, biases, step_sizes, momenta)
        self.calib_qubits = calib_qubits

    def initialize_biases(self):
        num_qubits = self.sampler.properties['num_qubits']
        biases = {"flux_biases": np.zeros(num_qubits), "anneal_offset": np.zeros(num_qubits)}
        gradients = {"flux_biases": np.zeros(num_qubits), "anneal_offset": np.zeros(num_qubits)}

    def make_sampler_kwags(self, biases):
        pass

