"""
@package docstring
Documentation for this module.
 
More details.
"""
import argparse


def parser_inv(parser):
    parser.add_argument(
        "--molecular",
        "-m",
        type=str,
        default="HH",
        help="Name of molecular.",
    )

    parser.add_argument(
        "--basis",
        "-b",
        type=str,
        default="cc-pv5z",
        help="Name of basis. We use cc-pv5z as default. Note we will remove core correlation of H atom; See https://github.com/pyscf/pyscf/issues/1795",
    )

    parser.add_argument(
        "--level",
        "-l",
        type=int,
        help="Level of DFT grid. Default is 4.",
        default=4,
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
        "--old_factor_scheme",
        "-fs",
        type=int,
        help="Scheme for old factor. Default is 1. -1 means use given old factor.",
        default=-1,
        choices=[-1, 1, 2, 3, 4],
    )

    parser.add_argument(
        "--old_factor",
        "-f",
        type=float,
        help="Old factor. Default is 0.9.",
        default=0.9,
    )

    parser.add_argument(
        "--error_inv",
        "-ei",
        type=float,
        help="Error for inversion. Default is 1e-6.",
        default=1e-6,
    )

    parser.add_argument(
        "--error_scf",
        "-es",
        type=float,
        help="Error for scf. Default is 1e-6.",
        default=1e-8,
    )

    parser.add_argument(
        "--inv_step",
        "-is",
        type=int,
        help="Number of steps for inversion. Default is 25000.",
        default=25000,
    )

    parser.add_argument(
        "--scf_step",
        "-ss",
        type=int,
        help="Number of steps for scf calculation. Default is 2500.",
        default=2500,
    )

    parser.add_argument(
        "--device",
        "-de",
        type=str,
        choices=["cpu", "cuda"],
        help="Device for inversion. Default is 'cuda'.",
        default="cuda",
    )

    parser.add_argument(
        "--noisy_print",
        "-n",
        type=bool,
        default=False,
        help="Whether to noisy print. Default is False.",
    )

    parser.add_argument(
        "--if_basis_str",
        "-bs",
        type=bool,
        default=False,
        help="Weather to use the basis set from basissetexchange. See https://www.basissetexchange.org. Default is False.",
    )

    parser.add_argument(
        "--load",
        type=bool,
        default=False,
        help="Weather to load the saved data. Default is False.",
    )

    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Weather to save the data. Default is False.",
    )

    parser.add_argument(
        "--method",
        "-me",
        type=str,
        help="Method for quantum chemistry calculation. Default is 'cisd'.",
        default="cisd",
    )
