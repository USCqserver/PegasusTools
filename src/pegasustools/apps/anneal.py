import argparse

from pegasustools.annealers.base import AnnealerModuleRunner, DWaveAnnealerModule, ScaledModule, MinorEmbeddingModule


def main(args=None):
    parser = argparse.ArgumentParser("Anneal")
    AnnealerModuleRunner.add_arguments(parser)
    DWaveAnnealerModule.add_arguments(parser)
    ScaledModule.add_arguments(parser)
    MinorEmbeddingModule.add_arguments(parser)
    args = parser.parse_args(args)
    kwargs_dict = vars(args)
    module = DWaveAnnealerModule(**kwargs_dict)
    module = ScaledModule(module, **kwargs_dict)
    module = MinorEmbeddingModule(module, **kwargs_dict)
    runner = AnnealerModuleRunner(module, **kwargs_dict)
    bqm, sampler, results = runner.main()

# class AnnealModule(ScaledModule, MinorEmbeddingModule):
#     pass
#
#
# def main(args=None):
#     parser = argparse.ArgumentParser()
#     AnnealModule.add_arguments(parser)
#     args = parser.parse_args(args)
#     annealer = AnnealModule(args)
#     bqm, sampler, results = annealer.main()


if __name__ == "__main__":
    main()
