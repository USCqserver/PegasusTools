import argparse
from pegasustools.annealers.dw import ScaledModule, MinorEmbeddingModule


class AnnealModule(ScaledModule, MinorEmbeddingModule):
    pass


def main(args=None):
    parser = argparse.ArgumentParser()
    AnnealModule.add_arguments(parser)
    args = parser.parse_args(args)
    annealer = AnnealModule(args)
    bqm, sampler, results = annealer.main()
    return bqm, sampler, results


if __name__ == "__main__":
    bqm, sampler, results = main()
