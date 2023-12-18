"""@package docstring
Documentation for this module.
 
More details.
"""
import argparse
import pyscf


HH = pyscf.M(
    atom=[["H", 1, 0, 0], ["H", -1, 0, 0]],
    unit="B",
)

HF = pyscf.M(
    atom=[["H", 1, 0, 0], ["F", -1, 0, 0]],
    unit="B",
)

Be = pyscf.M(
    atom=[["Be", 0, 0, 0]],
    unit="B",
)

Mol = {"H_2": HH, "HH": HH, "HF": HF, "Be": Be}


def get_args_quantum(parser: argparse.ArgumentParser):
    """Documentation for a function.

    More details.
    """
    parser.add_argument(
        "--level", "-l", metavar="LEVEL", type=int, help="level of atom grid", default=3
    )

    parser.add_argument(
        "--distance",
        "-d",
        type=float,
        help="Distance between atom H to the origin. Default is 1.0.",
        default=1.0,
    )

    parser.add_argument(
        "--rotate",
        "-r",
        type=str,
        help="Rotate the molecular. Can be x, y, z. Default is None",
        default=None,
    )

    parser.add_argument(
        "--qm_method",
        type=str,
        default="fci",
        help="Witch method we used to do the quantum chemical calculation.",
    )

    parser.add_argument(
        "--basis_set",
        type=str,
        default="aug-cc-pvdz",
        help="Witch basis set we used to do the quantum chemical calculation.",
    )

    parser.add_argument(
        "--molecular",
        type=str,
        default="H_2",
        help=f"Name of molecular. {list(Mol.keys())}",
    )

    parser.add_argument(
        "--qm_method_compare",
        type=str,
        default="b3lyp",
        help="Witch method we used to obtain the benchmark quantum chemical calculation."
        "This should be a dft level method.",
    )


def get_args_train(parser: argparse.ArgumentParser):
    """Documentation for a function.

    More details.
    """
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=5000, help="Number of epochs"
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=20,
        help="Batch size",
    )

    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-4,
        help="Learning rate",
        dest="lr",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="First_Run",
        help="Witch directory we save data to.",
    )

    parser.add_argument(
        "--scale", "-s", type=float, default=1, help="Downscaling factor of the images"
    )

    parser.add_argument(
        "--validation",
        "-v",
        type=int,
        default=20,
        dest="val",
        help="Number of the data that is used as validation",
    )

    parser.add_argument(
        "--training ",
        "-n",
        type=int,
        default=200,
        dest="train",
        help="Number of the data that is used as validation",
    )

    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )


def get_args_model(parser: argparse.ArgumentParser):
    """Documentation for a function.

    More details.
    """
    parser.add_argument(
        "--load",
        "-f",
        default=None,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )

    parser.add_argument(
        "--classes", "-c", type=int, default=1, help="Number of classes"
    )
    return parser.parse_args()
