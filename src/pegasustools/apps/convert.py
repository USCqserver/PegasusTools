import argparse
import pathlib

import dimod

from pegasustools.util.adj import read_ising_adjacency, save_ising_instance_graph
from dimod.serialization import coo
from dimod import lp, ConstrainedQuadraticModel, BinaryQuadraticModel
import pegasustools.util.lp as pgt_lp


def main():
    parser = argparse.ArgumentParser(description="Conversion between common Ising/QUBO formats")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("-i", choices=['coo', 'isn', 'qubo', 'lp'], required=True)
    parser.add_argument("-o", choices=['coo', 'isn', 'qubo', 'lp'], required=True)

    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file)
    output_file = pathlib.Path(args.output_file)
    if not input_file.is_file():
        print(f"File {input_file} does not exist.")
        exit(1)
    if args.i == 'coo':
        with open(input_file, 'r') as f:
            bqm = coo.load(f)
    elif args.i == 'isn':
        bqm = read_ising_adjacency(input_file, 1.0)
    elif args.i == 'qubo':
        bqm = read_ising_adjacency(input_file, 1.0, qubo=True)
    elif args.i == 'lp':
        bqm = lp.load(str(input_file))
        if bqm.num_constraints() > 0:
            print(f"Converting lp format with constraints is not supported")
            exit(1)
    else:
        raise ValueError(f"Invalid reading format {args.i}.")

    if args.o == 'coo':
        with open(output_file, 'w') as f:
            coo.dump(bqm, f)
    elif args.o == 'isn' or args.o=='qubo':
        print(f"Warning: output to isn/qubo is deprecated. Use coo instead")
        save_ising_instance_graph(dimod.to_networkx_graph(bqm), output_file)
    elif args.o == 'lp':
        bqm.relabel_variables({i: f"b{i}" for i in bqm.variables}, inplace=True)
        bqm.change_vartype(dimod.BINARY, inplace=True)
        cqm = ConstrainedQuadraticModel.from_quadratic_model(bqm)
        with open(output_file, 'w') as f:
            pgt_lp.dump(cqm, f)
    else:
        raise ValueError(f"Invalid writing format {args.o}.")


if __name__ == '__main__':
    main()