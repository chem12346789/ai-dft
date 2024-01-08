"""@package docstring
Documentation for this module.
 
More details.
"""
import logging
import gc
from pathlib import Path
import numpy as np
import pyscf
import argparse

from src.mrks_pyscf.mrksinv import Mrksinv
from src.mrks_pyscf.utils.mol import Mol, BASIS

path = Path(__file__).resolve().parents[1] / "data"
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)

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
    choices=[-1, 1, 2],
)

parser.add_argument(
    "--old_factor",
    "-f",
    type=float,
    help="Old factor. Default is 0.9.",
    default=0.9,
)

parser.add_argument(
    "--inv_step",
    "-s",
    type=int,
    help="Number of steps for inversion. Default is 25000.",
    default=25000,
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
    type=str,
    default="False",
    help="Whether to noisy print. Default is 'False'.",
)

parser.add_argument(
    "--method",
    "-me",
    type=str,
    choices=["cisd", "fci", "ccsd", "ccsdt", "hf"],
    help="Method for quantum chemistry calculation. Default is 'cisd'.",
    default="cisd",
)

args = parser.parse_args()

if args.old_factor_scheme == 1:
    from src.mrks_pyscf.utils.mol import old_function1 as old_function
elif args.old_factor_scheme == 2:
    from src.mrks_pyscf.utils.mol import old_function2 as old_function
else:

    def old_function(distance):
        """
        This function is used to determine the factor of mixing old and new density matrix in SCF process
        """
        return args.old_factor


if len(args.distance_list) == 3:
    distance_l = np.linspace(
        args.distance_list[0], args.distance_list[1], int(args.distance_list[2])
    )
else:
    distance_l = args.distance

molecular = Mol[args.molecular]

path_dir = path / f"data-{args.molecular}-{args.basis}-{args.method}-{args.level}"
if not path_dir.exists():
    path_dir.mkdir(parents=True)

logger = logging.getLogger(__name__)
logging.StreamHandler.terminator = ""
# clear the log
Path(path_dir / "inv.log").unlink(missing_ok=True)
logger.addHandler(logging.FileHandler(path_dir / "inv.log"))
logger.setLevel(logging.DEBUG)

for distance in distance_l:
    molecular[0][1] = distance
    logger.info("%s", f"The distance is {distance}.")

    basis = {}

    for i_atom in molecular:
        basis[i_atom[0]] = (
            BASIS[args.basis]
            if ((i_atom[0] == "H") and (args.basis in BASIS))
            else args.basis
        )

    mol = pyscf.M(
        atom=molecular,
        basis=basis,
    )

    mrks_inv = Mrksinv(
        mol,
        frac_old=old_function(distance),
        level=args.level,
        inv_step=args.inv_step,
        path=path_dir / f"{distance:.4f}",
        logger=logger,
        device=args.device,
        noisy_print=args.noisy_print,
    )

    mrks_inv.kernel(method=args.method)
    mrks_inv.inv_prepare()
    mrks_inv.inv()
    del mrks_inv, mol
    gc.collect()
    print("All done.\n")
