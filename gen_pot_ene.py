"""
@package docstring
Documentation for this module.
 
More details.
"""
import argparse
import logging
import gc
from pathlib import Path
import numpy as np
import pyscf

from mrks_pyscf.mrksinv import Mrksinv
from mrks_pyscf.utils.parser import parser_inv
from mrks_pyscf.utils.mol import Mol
from mrks_pyscf.utils.mol import old_function1
from mrks_pyscf.utils.mol import old_function2
from mrks_pyscf.utils.mol import old_function3
from mrks_pyscf.utils.mol import old_function4


path = Path(__file__).resolve().parents[1] / "data"
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
parser_inv(parser)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.StreamHandler.terminator = ""
if len(args.distance_list) == 3:
    distance_l = np.linspace(
        args.distance_list[0], args.distance_list[1], int(args.distance_list[2])
    )
    path_dir = path / f"data-{args.molecular}-{args.basis}-{args.method}-{args.level}"
    if not path_dir.exists():
        path_dir.mkdir(parents=True)
    Path(
        path_dir
        / f"inv_{args.distance_list[0]}_{args.distance_list[1]}_{args.distance_list[2]}.log"
    ).unlink(missing_ok=True)
    logger.addHandler(
        logging.FileHandler(
            path_dir
            / f"inv_{args.distance_list[0]}_{args.distance_list[1]}_{args.distance_list[2]}.log"
        )
    )
else:
    distance_l = args.distance
    path_dir = path / f"data-{args.molecular}-{args.basis}-{args.method}-{args.level}"
    if not path_dir.exists():
        path_dir.mkdir(parents=True)
    Path(path_dir / f"inv.log").unlink(missing_ok=True)
    logger.addHandler(logging.FileHandler(path_dir / f"inv.log"))
logger.setLevel(logging.DEBUG)

molecular = Mol[args.molecular]

for distance in distance_l:
    molecular[0][1] = distance
    logger.info("%s", f"The distance is {distance}.")

    if args.old_factor_scheme == 1:
        FRAC_OLD = old_function1(distance)
    elif args.old_factor_scheme == 2:
        FRAC_OLD = old_function2(distance)
    elif args.old_factor_scheme == 3:
        FRAC_OLD = old_function3(distance)
    elif args.old_factor_scheme == 4:
        FRAC_OLD = old_function4(distance)
    else:
        FRAC_OLD = args.old_factor

    mrks_inv = Mrksinv(
        molecular,
        path=path_dir / f"{distance:.4f}",
        args=args,
        logger=logger,
        frac_old=FRAC_OLD,
    )

    mrks_inv.kernel(method=args.method)
    mrks_inv.inv_prepare()
    mrks_inv.inv()
    del mrks_inv
    gc.collect()
    print("All done.\n")
