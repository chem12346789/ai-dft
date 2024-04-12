"""
@package docstring
Documentation for this module.
 
More details.
"""

import argparse
import gc
from pathlib import Path
import copy

from dft2cc.dft2cc import DFT2CC
from dft2cc.utils.parser import parser_inv
from dft2cc.utils.mol import Mol
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

for distance in distance_l:
    molecular = copy.deepcopy(Mol[args.molecular])
    molecular[0][1] = distance
    logger.info("%s", f"The distance is {distance}.")

    dft2cc = DFT2CC(
        molecular,
        path=path_dir / f"{distance:.4f}",
        args=args,
        logger=logger,
    )
    dft2cc.kernel(method=args.method)
    dft2cc.save_data()

    del dft2cc
    gc.collect()
    print("All done.\n")
