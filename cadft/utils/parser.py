"""
@package docstring
Documentation for this module.
 
More details.
"""

import argparse
from cadft.utils.mol import Mol


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
        type=bool,
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
        type=bool,
        default=True,
        help="Weather to use the basis set from basissetexchange. See https://www.basissetexchange.org. Default is False.",
    )

    parser.add_argument(
        "--cc_triple",
        type=bool,
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
        "--save",
        type=bool,
        default=False,
        help="Weather to save the data. Default is False.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100000,
        help="Number of epoch for training. Default is 100000.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Batch size for training. Default is 100000 (FCnet).",
    )

    parser.add_argument(
        "--adam",
        type=bool,
        action="store_true",
        default=True,
        help="Weather to use Adam optimizer. Default is True.",
    )

    parser.add_argument(
        "--eval_step",
        type=int,
        default=100,
        help="Step for evaluation. Default is 100.",
    )

    args = parser.parse_args()
    for i in range(len(args.extend_xyz)):
        args.extend_xyz[i] += 1
    return args
