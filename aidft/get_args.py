"""@package docstring
Documentation for this module.
 
More details.
"""

import argparse


def parser_model(parser: argparse.ArgumentParser):
    """
    Documentation for a function.

    More details.
    """
    parser.add_argument(
        "--molecular",
        "-m",
        type=str,
        default="HH",
        help="Name of molecular.",
    )

    parser.add_argument(
        "--basis",
        type=str,
        default="cc-pv5z",
        help="Name of basis. We use cc-pv5z as default."
        "Note we will remove core correlation of H atom; "
        "See https://github.com/pyscf/pyscf/issues/1795.",
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
        "--val_precent",
        "-vp",
        type=float,
        help="Percentage of validation data. Default is 0.1.",
        default=0.1,
    )

    parser.add_argument(
        "--level",
        "-l",
        type=int,
        help="Level of DFT grid. Default is 4.",
        default=4,
    )

    parser.add_argument(
        "--epochs",
        "-e",
        metavar="E",
        type=int,
        default=5000,
        help="Number of epochs",
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        metavar="B",
        type=int,
        default=20,
        help="Batch size",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="First_Run",
        help="Witch directory we save data to.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use mixed precision",
    )

    parser.add_argument(
        "--if_pad",
        default=True,
        help="If pad the data.",
    )

    parser.add_argument(
        "--if_flatten",
        default=False,
        help="If flatten the data.",
    )

    parser.add_argument(
        "--check",
        default=False,
        help="If check the data.",
    )

    parser.add_argument(
        "--division_epoch",
        type=int,
        default=50,
        dest="division_epoch",
        help="Division epoch. Default is 50. Validate every division_epoch.",
    )

    parser.add_argument(
        "--save_epoch",
        type=int,
        default=1000,
        dest="save_epoch",
        help="Save epoch. Default is 1000. Save every save_epoch.",
    )

    parser.add_argument(
        "--qm_method_compare",
        type=str,
        default="b3lyp",
        help="Witch method we used to obtain the benchmark quantum chemical calculation."
        "This should be a dft level method.",
    )

    parser.add_argument(
        "--old_factor",
        "-f",
        type=float,
        help="Old factor. Default is 0.9.",
        default=0.9,
    )

    parser.add_argument(
        "--old_factor_scheme",
        "-fs",
        type=int,
        help="Scheme for old factor. Default is 1. -1 means use given old factor.",
        default=-1,
        choices=[-1, 1, 2, 3, 4, 5],
    )

    parser.add_argument(
        "--scf_step",
        "-ss",
        type=int,
        help="Number of steps for scf calculation. Default is 2500.",
        default=2500,
    )

    parser.add_argument(
        "--error_scf",
        "-es",
        type=float,
        help="Error for scf. Default is 1e-6.",
        default=1e-8,
    )

    parser.add_argument(
        "--load",
        default=False,
        help="Load model from a .pth file",
    )

    parser.add_argument(
        "--ai_list",
        nargs="+",
        type=str,
        help="List of atom which we want to use AI to predict. Default is ['H'].",
        default=["H"],
    )

    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Use bilinear upsampling",
    )

    parser.add_argument(
        "--classes",
        "-c",
        type=int,
        default=1,
        help="Number of classes",
    )

    parser.add_argument(
        "--single",
        type=bool,
        default=False,
        help="Use single loop training method",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=[
            None,
            "sgd",
            "rmsprop",
            "adam",
            "adadelta",
            "adamax",
            "radam",
            "nadam",
        ],
        help="Witch model we used to do the machine learning.",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=[
            None,
            "plateau",
            "cosine",
            "step",
            "none",
        ],
        help="Witch model we used to do the machine learning.",
    )

    parser.add_argument(
        "--noisy_print",
        "-n",
        type=bool,
        default=False,
        help="Whether to noisy print. Default is False.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay. Default is 0.",
        default=0,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        help="Momentum. Default is 0.99.",
        default=0.99,
    )

    parser.add_argument(
        "--gradient_clipping",
        type=float,
        help="Gradient clipping. Default is 1.0.",
        default=1.0,
    )

    return parser.parse_args()
