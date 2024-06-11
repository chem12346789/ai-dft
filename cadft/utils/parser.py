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
        type=int,
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
        "--epoch",
        type=int,
        default=10000,
        help="Number of epoch for training. Default is 10000.",
    )

    parser.add_argument(
        "--ene_grid_factor",
        type=float,
        default=0,
        help="Weather to use the energy grid label. 0 for not using. Default is 0.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Batch size for training. Default is 100000 (FCnet).",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=100,
        help="Number of hidden size for training. Default is 100.",
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
        "--eval_step",
        type=int,
        default=100,
        help="Step for evaluation. Default is 100.",
    )

    parser.add_argument(
        "--noise_print",
        type=str2bool,
        default=False,
        help="Weather to print the noise. Default is False.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Precision for the training. Default is float64.",
    )

    args = parser.parse_args()
    for i in range(len(args.extend_xyz)):
        args.extend_xyz[i] += 1

    return args
