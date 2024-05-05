from pathlib import Path
from itertools import product

import h5py
import numpy as np

from cadft.utils import Mol

ATOM_LIST = [
    "H",
    "C",
]
ATOM_STR_DICT = [
    "Methane",
    "Ethane",
    "Ethylene",
    "Acetylene",
    "Allene",
    "Cyclopropene",
    "Propyne",
    "Cyclopropane",
    "Propylene",
    "Propane",
    "Isobutane",
    "Butane",
    "Butadiene",
    "Butyne",
    "Bicyclobutane",
    "Cyclopropylmethyl",
    "Cyclobutane",
    "Spiropentane",
    "Benzene",
    "Pentane",
    "Isopentane",
    "Neopentane",
    "Cyclopentane",
]


def numpy_to_hdf5():
    path = Path("./") / "data"
    path_h5py = Path("./") / "data" / "file.h5"
    with h5py.File(path_h5py, "w") as f:
        for i_atom in ATOM_LIST:
            for j_atom in ATOM_LIST:
                atom_name = i_atom + j_atom
                grp = f.require_group(atom_name)
                for i_molecular in ATOM_STR_DICT:
                    dset = grp.require_group(i_molecular)
                    for (
                        extend_atom,
                        extend_xyz,
                        distance,
                        (if_sub, magic_str),
                    ) in product(
                        [0, 1],
                        [1, 2, 3],
                        np.linspace(-0.5, 0.5, 41),
                        [
                            (False, "weight/energy_nuc"),
                            (False, "weight/e_ccsd"),
                            (True, f"{atom_name}/input/input"),
                            (True, f"{atom_name}/output/output_dm1"),
                            (True, f"{atom_name}/output/output_exc"),
                        ],
                    ):

                        if abs(distance) < 1e-3:
                            if (extend_atom != 0) or extend_xyz != 1:
                                print(
                                    f"Skip {i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}"
                                )
                                continue

                        print(
                            f"{atom_name}/{magic_str}_{extend_atom}_{magic_str}_{i_molecular}_{extend_xyz}_{distance:.4f}"
                        )
                        if if_sub:
                            molecular = Mol[i_molecular]
                            natom = len(molecular)
                            for i, j in product(range(natom), range(natom)):
                                if molecular[i][0] != i_atom:
                                    continue
                                if molecular[j][0] != j_atom:
                                    continue

                                data = np.load(
                                    path
                                    / f"{magic_str}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}_{i}_{j}.npy"
                                )
                                dset.create_dataset(
                                    f"{magic_str}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}_{i}_{j}",
                                    data=data,
                                )
                        else:
                            data = np.load(
                                path
                                / f"{magic_str}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}.npy"
                            )
                            dset.create_dataset(
                                f"{magic_str}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                                data=data,
                            )


numpy_to_hdf5()
