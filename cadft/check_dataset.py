from itertools import product

import argparse
import torch
import numpy as np

from cadft.utils import add_args, DataBase


ATOM_LIST = [
    "H",
    "C",
]
TRAIN_STR_DICT = [
    "Methane",
    # "Ethane",
    # "Ethylene",
    # "Acetylene",
    # "Allene",
    # "Cyclopropene",
    "Propyne",
    # "Cyclopropane",
    # "Propylene",
    # "Propane",
    # "Isobutane",
    # "Butane",
    # "Butadiene",
    # "Butyne",
    # "Bicyclobutane",
    # "Cyclopropylmethyl",
    # "Cyclobutane",
    # "Spiropentane",
    # "Benzene",
    # "Pentane",
    # "Isopentane",
    # "Neopentane",
    # "Cyclopentane",
]

parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

database_train = DataBase(args, ATOM_LIST, TRAIN_STR_DICT, device, normalize=False)
# database_eval = DataBase(args, ATOM_LIST, EVAL_STR_DICT, device)

max_value1 = 0
max_value2 = 0
max_value3 = 0
max_value4 = 0

for i_atom, j_atom in product(ATOM_LIST, ATOM_LIST):
    atom_name = i_atom + j_atom
    for i_keys in database_train.input[atom_name].keys():
        max_value1 = max(
            np.mean(
                np.abs(
                    (
                        database_train.middle[atom_name][i_keys]
                        / (np.cosh((database_train.input[atom_name][i_keys])) - 0.95)
                        + 1e-5
                        + 1e-5
                        * np.random.randn(
                            *database_train.input[atom_name][i_keys].shape
                        )
                    )
                    * (np.cosh((database_train.input[atom_name][i_keys])) - 0.95)
                    - database_train.middle[atom_name][i_keys]
                )
            ),
            max_value1,
        )
        max_value2 = max(
            np.mean(
                np.abs(
                    1e-5
                    + 1e-5
                    * np.random.randn(*database_train.input[atom_name][i_keys].shape)
                )
            ),
            max_value2,
        )
        max_value3 = max(
            np.max(
                database_train.middle[atom_name][i_keys]
                / (np.cosh((database_train.input[atom_name][i_keys])) - 0.95)
            ),
            max_value3,
        )
        max_value4 = max(
            np.max(database_train.middle[atom_name][i_keys]),
            max_value4,
        )
print()
print(max_value1, max_value2, max_value3, max_value4)
