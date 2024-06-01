from pathlib import Path
from itertools import product

import torch
import numpy as np

from cadft.utils.BasicDataset import BasicDataset
from cadft.utils.mol import Mol

# perprocess factor, can be imported from other files.
MIDDLE_SCALE, OUTPUT_SCALE = 1000.0, 1000.0


def gen_logger(distance_list):
    """
    Function to distance list and generate logger
    """
    if len(distance_list) == 3:
        distance_l = np.linspace(
            distance_list[0], distance_list[1], int(distance_list[2])
        )
    else:
        distance_l = distance_list
    return distance_l


class DataBase:
    """Documentation for a class."""

    def __init__(
        self,
        molecular_list,
        extend_atom,
        extend_xyz,
        distance_list,
        batch_size,
        ene_grid_factor,
        device,
    ):
        self.molecular_list = molecular_list
        self.extend_atom = extend_atom
        self.extend_xyz = extend_xyz
        self.distance_list = distance_list
        self.batch_size = batch_size
        self.ene_grid_factor = ene_grid_factor
        self.device = device

        self.distance_l = gen_logger(self.distance_list)
        self.data = {}
        self.data_gpu = {}
        self.ene = {}
        self.shape = {}

        self.name_list = []

        for (
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
        ) in product(
            self.molecular_list,
            self.extend_atom,
            self.extend_xyz,
            self.distance_l,
        ):
            name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            if abs(distance) < 1e-3:
                if (extend_atom != 0) or extend_xyz != 1:
                    print(f"Skip: {name:>40}")
                    continue

            if extend_atom >= len(Mol[name_mol]):
                print(f"Skip: {name:>40}")
                continue

            self.name_list.append(name)

        for name in self.name_list:
            if not (Path("data/grids/") / f"data_{name}.npz").exists():
                print(f"No file: {name:>40}")
                continue
            self.load_data(name)

    def load_data(self, name):
        """
        Load the data.
        """
        data = np.load(Path("data/grids/") / f"data_{name}.npz")

        weight = data["weights"]
        input_mat = data["rho_dft"]
        middle_mat = (data["rho_cc"] - data["rho_dft"]) * MIDDLE_SCALE
        ene = data["delta_ene_dft"] * OUTPUT_SCALE

        if (self.ene_grid_factor) and ("exc_over_dm_cc_grids" in data.files):
            output_mat = data["exc_over_dm_cc_grids"] * OUTPUT_SCALE
        else:
            output_mat = np.array([])

        self.ene[name] = ene
        self.shape[name] = input_mat.shape
        input_ = {}
        middle_ = {}
        weight_ = {}
        output_ = {}

        for i_atom in range(input_mat.shape[0]):
            for i in range(input_mat.shape[1]):
                key_ = f"{i_atom}_{i}"
                input_[key_] = input_mat[i_atom, i, :]
                middle_[key_] = middle_mat[i_atom, i, :]
                weight_[key_] = weight[i_atom, i, :]
                if output_mat.size != 0:
                    output_[key_] = output_mat[i_atom, i, :]
                else:
                    output_[key_] = np.array([])

        self.data_gpu[name] = BasicDataset(
            input_, middle_, output_, weight_, self.batch_size
        ).load_to_gpu()
        self.data[name] = {"input": input_, "middle": middle_, "weight": weight_}

        print(
            f"Load {name:>30}, mean input: {np.mean(input_mat):7.4f}, mean middle: {np.mean(middle_mat):7.4f}, mean output: {ene:7.4f}."
        )
