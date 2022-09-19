import argparse
from dwave.preprocessing import ClipComposite
from pegasustools.qac import PegasusQACEmbedding, AbstractQACEmbedding
from pegasustools.nqac import PegasusNQACEmbedding, PegasusK4NQACGraph
from pegasustools.annealers.base import CompositeAnnealerModule, DWaveAnnealerModule, ScaledModule, \
    AnnealerModuleRunner, MinorEmbeddingModule, VariableMappingModule


class QACAnnealModule(CompositeAnnealerModule):
    def __init__(self, child_module, qac_method='qac', qac_penalty=0.1, qac_scale=1.0, qac_mode='qac',
                           qac_clip=None, **kwargs):
        super(QACAnnealModule, self).__init__(child_module, **kwargs)
        self.qac_method = qac_method
        self.qac_penalty = qac_penalty
        self.qac_scale = qac_scale
        self.qac_mode = qac_mode
        self.qac_clip = abs(qac_clip) if qac_clip is not None else None

    @classmethod
    def add_arguments(cls, parser):
        p = parser.add_argument_group("QAC")
        p.add_argument("--qac-method", default="qac", choices=["qac", "k4"])
        p.add_argument("--qac-penalty", type=float, default=0.1,
                       help="Penalty strength for QAC")
        p.add_argument("--qac-scale", type=float, default=1.0,
                       help="Scale factor for logical couplings for QAC")
        p.add_argument("--qac-mode", type=str, choices=['qac', 'c', 'all'], default='qac')
        p.add_argument("--qac-clip", type=float,
                       help="Clip the absolute value of the physical couplings after embedding QAC"
                       )

    def initialize_sampler(self) -> AbstractQACEmbedding:
        sampler = self.child_module.initialize_sampler()
        if self.verbose:
            print(f"QAC Penalty: {self.qac_penalty}")
            print(f"QAC Problem scale: {self.qac_scale}")
        qac_args = {
            "qac_penalty_strength": self.qac_penalty,
            "qac_problem_scale": self.qac_scale,
            "qac_decoding": self.qac_mode,
            "qac_clip": self.qac_clip
        }
        if self.qac_method == "qac":
            qac_sampler = PegasusQACEmbedding(16, sampler)
            qac_graph = qac_sampler.qac_graph
        elif self.qac_method == "k4":
            if self.qac_mode == 'c':
                k2_mode = True
            else:
                k2_mode = False
            qac_graph = PegasusK4NQACGraph.from_sampler(16, sampler, k2_mode=k2_mode)
            qac_sampler = PegasusNQACEmbedding(16, sampler, qac_graph)
        else:
            raise RuntimeError(f"Invalid method {self.qac_method}")

        self._sampler_kwargs = qac_args
        return qac_sampler


def main(args=None):
    parser = argparse.ArgumentParser("Anneal")
    AnnealerModuleRunner.add_arguments(parser)
    DWaveAnnealerModule.add_arguments(parser)
    QACAnnealModule.add_arguments(parser)
    MinorEmbeddingModule.add_arguments(parser)
    VariableMappingModule.add_arguments(parser)
    ScaledModule.add_arguments(parser)
    args = parser.parse_args(args)
    kwargs_dict = vars(args)
    module = DWaveAnnealerModule(**kwargs_dict)
    module = QACAnnealModule(module, **kwargs_dict)
    module = MinorEmbeddingModule(module, **kwargs_dict)
    module = VariableMappingModule(module, **kwargs_dict)
    module = ScaledModule(module, **kwargs_dict)

    preview_columns = ['energy', 'error_p', 'tie_p', 'chain_break_fraction', 'rep', 'num_occurrences']
    runner = AnnealerModuleRunner(module, preview_columns=preview_columns, **kwargs_dict)
    bqm, sampler, results = runner.main()


if __name__ == "__main__":
    main()
