"""
@package docstring
Documentation for this module.
 
More details.
"""

import argparse
from cadft.utils.mol import Mol


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True" "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False" "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_args(parser: argparse.ArgumentParser):
    """
    Documentation for a function.

    More details.
    """
    parser.add_argument(
        "--name_mol",
        "-m",
        nargs="+",
        type=str,
        default="HH",
        help=f"Name of molecular. Must in {list(Mol.keys())}.",
    )

    parser.add_argument(
        "--distance_list",
        "-dl",
        nargs="+",
        type=float,
        help="Distance between atom H to the origin. Default is 1.0.",
        default=1.0,
    )

    parser.add_argument(
        "--extend_atom",
        type=str,
        nargs="+",
        default=0,
        help="Number of atoms to extend. Default is 0.",
    )

    parser.add_argument(
        "--extend_xyz",
        type=int,
        nargs="+",
        default=0,
        help="Number of xyz to extend. 0 for x, 1 for y, 2 for z. Default is 0.",
    )

    parser.add_argument(
        "--noisy_print",
        "-n",
        type=str2bool,
        default=False,
        help="Whether to noisy print. Default is False.",
    )

    parser.add_argument(
        "--basis",
        "-b",
        type=str,
        default="cc-pv5z",
        help="Name of basis. We use cc-pv5z as default. Note we will remove core correlation of H atom; See https://github.com/pyscf/pyscf/issues/1795",
    )

    parser.add_argument(
        "--if_basis_str",
        "-bs",
        type=str2bool,
        default=True,
        help="Weather to use the basis set from basissetexchange. See https://www.basissetexchange.org. Default is False.",
    )

    parser.add_argument(
        "--cc_triple",
        type=str2bool,
        default="False",
        help="Weather to use the noniterative CCSD(T) in the coupled cluster method. Default is False.",
    )

    # for machine learning

    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Weather to load the saved data. Default is " ".",
    )

    parser.add_argument(
        "--load_epoch",
        type=int,
        default=-1,
        help="Epoch to load the model, -1 for the last epoch. Default is -1.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=10000,
        help="Number of epoch for training. Default is 10000.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Batch size for training. Default is 100000 (FCnet).",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=100,
        help="Number of input size for training. Default is 100.",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=100,
        help="Number of hidden size for training. Default is 100.",
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=100,
        help="Number of output size for training. Default is 100.",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of layers for training. Default is 3.",
    )

    parser.add_argument(
        "--residual",
        type=int,
        default=0,
        help="Type of residual for training. Default is 0. 0 for no residual, 1 for residual, 2 for multi-level residual.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Precision for the training. Default is float64.",
    )

    parser.add_argument(
        "--ene_weight",
        type=float,
        default=1.0,
        help="Weight for the energy. Default is 1.0.",
    )

    parser.add_argument(
        "--pot_weight",
        type=float,
        default=1.0,
        help="Weight for the potential. Default is 1.0.",
    )

    parser.add_argument(
        "--with_eval",
        type=str2bool,
        default=True,
        help="Weather to use the reduce on plateau for the learning rate. Default is True. This will use the data from the eval set.",
    )

    parser.add_argument(
        "--eval_step",
        type=int,
        default=100,
        help="Step for evaluation. Default is 100.",
    )

    parser.add_argument(
        "--train_atom_list",
        nargs="+",
        type=str,
        help="List of atoms to train. Default is H and C atoms.",
        default=["H", "C"],
    )

    parser.add_argument(
        "--load_inv",
        type=str2bool,
        default=False,
        help="Weather to load the inversed potential. Default is False.",
    )

    # for test
    parser.add_argument(
        "--from_data",
        type=str2bool,
        default=False,
        help="Weather to use the data from the data file. Default is False.",
    )

    parser.add_argument(
        "--require_grad",
        type=str2bool,
        default=False,
        help="Weather to require the grad. Default is False.",
    )

    args = parser.parse_args()
    for i in range(len(args.extend_xyz)):
        args.extend_xyz[i] += 1

    return args
