from pathlib import Path
import copy
from itertools import product

import pyscf
from pyscf import dft
import torch
import numpy as np

from cadft.utils.logger import gen_logger
from cadft.utils.nao import NAO
from cadft.utils.mol import Mol
import cadft


class DataBase:
    """Documentation for a class."""

    def __init__(
        self,
        args,
        molecular_list,
        device,
    ):
        self.args = args
        self.molecular_list = molecular_list
        self.device = device

        data_path = Path("data")
        self.dir_grids = data_path / "grids/"
        self.dir_weight = data_path / "weight/"

        self.distance_l = gen_logger(args.distance_list)
        self.data = {}
        self.input = {}
        self.middle = {}
        self.output = {}

        for (
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
        ) in product(
            self.molecular_list,
            self.args.extend_atom,
            self.args.extend_xyz,
            self.distance_l,
        ):
            if abs(distance) < 1e-3:
                if (extend_atom != 0) or extend_xyz != 1:
                    print(
                        f"Skip: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}"
                    )
                    continue

            if extend_atom >= len(Mol[name_mol]):
                print(f"Skip: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}")
                continue

            name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            if not (self.dir_grids / f"data_{name}.npz").exists():
                print(
                    f"No file: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}"
                )
                continue

            self.load_data(name)

    def load_data(self, name):
        """
        Load the data.
        """
        data = np.load(self.dir_grids / f"data_{name}.npz")

        e_cc = np.load(self.dir_weight / f"e_ccsd_{name}.npy")
        e_dft = np.load(self.dir_weight / f"e_dft_{name}.npy")

        input_mat = data["rho_dft"]
        middle_mat = data["rho_cc"]
        output_mat = data["exc_over_dm_cc_grids"]

        self.data[name] = {
            "e_cc": e_cc,
            "e_dft": e_dft,
        }

        for i in range(input_mat.shape[1]):
            self.input[f"{name}_{i}"] = input_mat[:, i]
            self.middle[f"{name}_{i}"] = middle_mat[:, i] - input_mat[:, i]
            self.output[f"{name}_{i}"] = output_mat[i, np.newaxis]

    def check(self, model_list=None, if_equilibrium=True):
        """
        Check the input data, if model_list is not none, check loss of the model.
        """
        ene_loss = []
        ene_loss_1 = []
        ene_loss_2 = []
        rho_loss = []
        dip_x_loss = []
        dip_y_loss = []
        dip_z_loss = []
        name_train = []
