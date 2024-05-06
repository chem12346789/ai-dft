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
        grp_weight = f.require_group("weight")
        for i_molecular in ATOM_STR_DICT:
            for (
                extend_atom,
                extend_xyz,
                distance,
            ) in product(
                [0, 1],
                [1, 2, 3],
                np.linspace(-0.5, 0.5, 41),
            ):
                if abs(distance) < 1e-3:
                    if (extend_atom != 0) or extend_xyz != 1:
                        print(
                            f"Skip {i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}"
                        )
                        continue

                for magic_str in [
                    "energy_nuc",
                    "e_ccsd",
                ]:
                    data = np.load(
                        path
                        / "weight"
                        / f"{magic_str}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}.npy"
                    )
                    grp_weight.create_dataset(
                        f"{magic_str}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                        data=data,
                    )

                for i_atom, j_atom, (dir_name, file_name) in product(
                    ATOM_LIST,
                    ATOM_LIST,
                    [
                        ("input", "input"),
                        ("output", "output_dm1"),
                        ("output", "output_exc"),
                    ],
                ):
                    atom_name = i_atom + j_atom
                    grp = f.require_group(atom_name).require_group(dir_name)

                    print(
                        f"{atom_name}/{dir_name}/{file_name}_{extend_atom}_{i_molecular}_{extend_xyz}_{distance:.4f}"
                    )
                    molecular = Mol[i_molecular]
                    natom = len(molecular)
                    for i, j in product(range(natom), range(natom)):
                        if molecular[i][0] != i_atom:
                            continue
                        if molecular[j][0] != j_atom:
                            continue
                        data = np.load(
                            path
                            / atom_name
                            / dir_name
                            / f"{file_name}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}_{i}_{j}.npy"
                        )
                        grp.create_dataset(
                            f"{file_name}_{i_molecular}_{extend_atom}_{extend_xyz}_{distance:.4f}_{i}_{j}",
                            data=data,
                        )


numpy_to_hdf5()
