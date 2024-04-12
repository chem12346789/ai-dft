"""@package docstring
Documentation for this module.
 
More details.
"""
import argparse
from pathlib import Path
import numpy as np

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

molecular = Mol[args.molecular]
distance_l, logger, path_dir = gen_logger(
    args.distance_list,
    f"{args.molecular}-{args.basis}-{args.method}-{args.level}",
    path,
)

for distance in distance_l:
    molecular[0][1] = distance
    logger.info("%s", f"The distance is {distance}.")
    FRAC_OLD = old_function(distance, args.old_factor_scheme, args.old_factor)

    mrks_inv = DFT2CC(
        molecular,
        path=path_dir / f"{distance:.4f}",
        args=args,
        logger=logger,
        frac_old=FRAC_OLD,
    )

    weight_grid = mrks_inv.grids.vector_to_matrix(mrks_inv.grids.weights)
    weight_check = mrks_inv.grids.matrix_to_vector(weight_grid)
    mrks_inv.logger.info(
        f"{np.linalg.norm(weight_check - mrks_inv.grids.weights):16.10f}\n"
    )
    np.save(mrks_inv.path / "weight.npy", weight_grid)
