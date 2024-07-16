from pathlib import Path
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from cadft.utils.mol import Mol, HASH_LIST, ATOM_HASH_DICT
from cadft.utils.env_var import DATA_PATH


def process(data, dtype):
    """
    Load the whole data to the gpu.
    """
    if len(data.shape) == 4:
        return data.to(
            device="cuda",
            dtype=dtype,
            memory_format=torch.channels_last,
        )
    else:
        return data.to(
            device="cuda",
            dtype=dtype,
        )


class BasicDataset:
    """
    Documentation for a class.
    """

    def __init__(self, input_, middle_, output_, weight_, batch_size, dtype):
        self.input = input_
        self.middle = middle_
        self.output = output_
        self.weight = weight_
        self.ids = list(input_.keys())
        self.batch_size = batch_size
        if dtype == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input": self.input[self.ids[idx]],
            "middle": self.middle[self.ids[idx]],
            "output": self.output[self.ids[idx]],
            "weight": self.weight[self.ids[idx]],
        }

    def load_to_gpu(self):
        """
        Load the whole data to the device.
        """
        dataloader = DataLoader(
            self,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
        )

        dataloader_gpu = []
        for batch in dataloader:
            batch_gpu = {}
            # move images and labels to correct device and type
            (
                batch_gpu["input"],
                batch_gpu["middle"],
                batch_gpu["output"],
                batch_gpu["weight"],
            ) = (
                process(batch["input"], self.dtype),
                process(batch["middle"], self.dtype),
                process(batch["output"], self.dtype),
                process(batch["weight"], self.dtype),
            )
            dataloader_gpu.append(batch_gpu)
        return dataloader_gpu


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
        input_size,
        basis,
        batch_size,
        device,
        dtype,
    ):
        self.molecular_list = molecular_list
        self.extend_atom = extend_atom
        self.extend_xyz = extend_xyz
        self.distance_list = distance_list
        self.input_size = input_size
        self.basis = basis
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.distance_l = gen_logger(self.distance_list)

        self.name_list = []
        self.rng = np.random.default_rng()
        self.data_gpu = {}
        self.shape = {}

        for hash_char in HASH_LIST:
            self.data_gpu[hash_char] = {}

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
            name = f"{name_mol}_{self.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            if abs(distance) < 1e-3:
                if (extend_atom != 0) or extend_xyz != 1:
                    print(f"Skip: {name:>40}")
                    continue

            if extend_atom >= len(Mol[name_mol]):
                print(f"Skip: {name:>40}")
                continue

            if not (Path(f"{DATA_PATH}") / f"data_{name}.npz").exists():
                print(f"No file: {name:>40}")
                continue

            self.name_list.append(name)
            self.load_data(name, Mol[name_mol])

    def load_data(self, name, mol):
        """
        Load the data.
        """
        data = np.load(Path(f"{DATA_PATH}") / f"data_{name}.npz")

        weight = data["weights"]
        # input_mat = data["rho_inv"]
        input_mat = data["rho_inv_4"]
        middle_mat = data["vxc_b3lyp"]
        output_mat = data["exc1_tr_b3lyp"]

        self.shape[name] = output_mat.shape
        input_ = {}
        middle_ = {}
        weight_ = {}
        output_ = {}

        for hash_char in HASH_LIST:
            input_[hash_char] = {}
            middle_[hash_char] = {}
            weight_[hash_char] = {}
            output_[hash_char] = {}

        for i_atom in range(self.shape[name][0]):
            hash_ = ATOM_HASH_DICT[mol[i_atom][0]]
            if self.input_size == 1:
                input_[hash_][i_atom] = input_mat[0, i_atom, :, :][np.newaxis, :, :]
            elif self.input_size == 4:
                input_[hash_][i_atom] = input_mat[:, i_atom, :, :]
            else:
                raise ValueError("input_size should be 1 or 4.")

            middle_[hash_][i_atom] = middle_mat[i_atom, :, :][np.newaxis, :, :]
            weight_[hash_][i_atom] = weight[i_atom, :, :][np.newaxis, :, :]
            output_[hash_][i_atom] = output_mat[i_atom, :, :][np.newaxis, :, :]

            print(
                f"Load: {name:>30}, atom: {mol[i_atom][0]:>3}, "
                f"key_: {i_atom:>3}, hash_: {hash_}, \n"
                f"mean input: {np.mean(input_[hash_][i_atom]):>7.4e}, "
                f"max input: {np.max(input_[hash_][i_atom]):>7.4e}, "
                f"var input: {np.var(input_[hash_][i_atom]):>7.4e}, \n"
                f"mean middle: {np.mean(middle_[hash_][i_atom]):>7.4e}, "
                f"max middle: {np.max(np.abs(middle_[hash_][i_atom])):>7.4e}, "
                f"var middle: {np.var(middle_[hash_][i_atom]):>7.4e}, \n"
                f"mean output: {np.mean(output_[hash_][i_atom]):>7.4e}, "
                f"max output: {np.max(np.abs(output_[hash_][i_atom])):>7.4e} "
                f"var output: {np.var(output_[hash_][i_atom]):>7.4e}\n"
            )

        for hash_char in HASH_LIST:
            self.data_gpu[hash_char][name] = BasicDataset(
                input_[hash_char],
                middle_[hash_char],
                output_[hash_char],
                weight_[hash_char],
                self.batch_size,
                self.dtype,
            ).load_to_gpu()
