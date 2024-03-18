"""
@package docstring
Documentation for this module.
 
More details.
"""

import argparse
import gc
from pathlib import Path

from mrks_pyscf.mrksinv import Mrksinv
from mrks_pyscf.utils.parser import parser_inv
from mrks_pyscf.utils.mol import Mol
from mrks_pyscf.utils.mol import old_function
from mrks_pyscf.utils.logger import gen_logger


path = Path(__file__).resolve().parents[1] / "data"
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
parser_inv(parser)
args = parser.parse_args()

distance_l, logger, path_dir = gen_logger(
    args.distance_list,
    f"{args.molecular}-{args.basis}-{args.method}-{args.level}",
    path,
)
molecular = Mol[args.molecular]

for distance in distance_l:
    molecular[0][1] = distance
    logger.info("%s", f"The distance is {distance}.")
    FRAC_OLD = old_function(distance, args.old_factor_scheme, args.old_factor)

    mrks_inv = Mrksinv(
        molecular,
        path=path_dir / f"{distance:.4f}",
        args=args,
        logger=logger,
        frac_old=FRAC_OLD,
    )

    if args.load:
        mrks_inv.load_prepare_inverse()
    else:
        mrks_inv.kernel(method=args.method)
        if args.save:
            mrks_inv.save_prepare_inverse()
    mrks_inv.inv()
    mrks_inv.save_mol_info()
    mrks_inv.save_b3lyp()

    del mrks_inv
    gc.collect()
    print("All done.\n")
