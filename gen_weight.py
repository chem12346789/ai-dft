"""@package docstring
Documentation for this module.
 
More details.
"""
import argparse
import logging
import gc
from pathlib import Path
import numpy as np
import pyscf

from src.mrks_pyscf.mrksinv import Mrksinv
from src.mrks_pyscf.utils.parser import parser_inv
from src.mrks_pyscf.utils.mol import Mol
from src.mrks_pyscf.utils.mol import old_function1
from src.mrks_pyscf.utils.mol import old_function2
from src.mrks_pyscf.utils.mol import old_function3
from src.mrks_pyscf.utils.mol import old_function4


path = Path(__file__).resolve().parents[1] / "data"
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
parser_inv(parser)
args = parser.parse_args()

mrks_inv = Mrksinv(
    molecular,
    path=path_dir / f"{distance:.4f}",
    args=args,
    logger=logger,
    frac_old=FRAC_OLD,
)
