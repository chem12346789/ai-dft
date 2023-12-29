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
    "--frac_old",
    "-f",
    type=float,
    help="Used for SCF process to determine the fraction of old density matrix. Default is 0.8.",
    default=0.8,
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
    "--inv_change_vj",
    "-v",
    type=bool,
    help="Whether to change the vj matrix. Default is False.",
    default=False,
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
    "--method",
    "-me",
    type=str,
    choices=["cisd", "fci", "ccsd", "ccsdt"],
    help="Method for quantum chemistry calculation. Default is 'cisd'.",
    default="DFT",
)

args = parser.parse_args()

if len(args.distance_list) == 3:
    distance_l = np.linspace(
        args.distance_list[0], args.distance_list[1], int(args.distance_list[2])
    )
else:
    distance_l = args.distance

molecular = Mol[args.molecular]

if args.inv_change_vj:
    path_dir = path / f"data-vj-{args.molecular}-{args.basis}-{args.method}-{args.level}"
else:
    path_dir = path / f"data-{args.molecular}-{args.basis}-{args.method}-{args.level}"
if not path_dir.exists():
    path_dir.mkdir(parents=True)

logger = logging.getLogger(__name__)
logging.StreamHandler.terminator = ""
# clear the log
Path(path_dir / "inv.log").unlink(missing_ok=True)
logger.addHandler(logging.FileHandler(path_dir / "inv.log"))
logger.setLevel(logging.DEBUG)

for coordinate in distance_l:
    molecular[0][1] = coordinate
    logger.info("%s", f"The distance is {coordinate}.")

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
        frac_old=args.frac_old,
        level=args.level,
        inv_step=args.inv_step,
        path=path_dir / f"{coordinate:.4f}",
        logger=logger,
        inv_change_vj=args.inv_change_vj,
        device=args.device,
    )

    mrks_inv.kernel(method=args.method)
    mrks_inv.inv_prepare()
    mrks_inv.inv()
    del mrks_inv, mol
    gc.collect()
    print("All done.\n")
