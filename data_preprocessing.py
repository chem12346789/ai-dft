"""@package docstring
Documentation for this module.
 
More details.
"""

import gc
from pathlib import Path
import numpy as np
import pyscf

from src.mrks_pyscf.mrksinv import Mrksinv

path = Path(__file__).resolve().parents[1] / "data"

for coorinate in np.linspace(0.5, 3, 51):
    mol = pyscf.M(
        atom=[["H", coorinate, 0, 0], ["H", -coorinate, 0, 0]],
        basis="aug-cc-pvqz",
        unit="B",
    )
    print(f"The distance is {coorinate}.")

    MOLECULAR_NAME = ""
    for i in mol.atom:
        MOLECULAR_NAME += i[0]

    mrks_inv = Mrksinv(
        mol,
        frac_old=0.8,
        level=4,
        path=path / f"data-{MOLECULAR_NAME}" / f"{coorinate:.2f}",
    )
    mrks_inv.kernel(method="cisd")
    mrks_inv.add_kin()
    del mrks_inv, mol
    gc.collect()
    print("All done.\n")
