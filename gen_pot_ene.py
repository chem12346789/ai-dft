"""
@package docstring
Documentation for this module.
 
More details.
"""

import argparse
import gc
from pathlib import Path

from dft2cc.dft2cc import DFT2CC
from dft2cc.utils.parser import parser_inv
from dft2cc.utils.mol import Mol
from dft2cc.utils.mol import old_function
from dft2cc.utils.logger import gen_logger


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

    dft2cc = DFT2CC(
        molecular,
        path=path_dir / f"{distance:.4f}",
        args=args,
        logger=logger,
        frac_old=FRAC_OLD,
    )

    if args.load:
        dft2cc.load_prepare_inverse()
    else:
        dft2cc.kernel(method=args.method)
        if args.save:
            dft2cc.save_prepare_inverse()

    dft2cc.inv()
    dft2cc.save_mol_info()
    dft2cc.save_b3lyp()

    del dft2cc
    gc.collect()
    print("All done.\n")
