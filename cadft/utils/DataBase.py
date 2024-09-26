from pathlib import Path
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from cadft.utils.mol import Mol
from cadft.utils.env_var import DATA_PATH

AU2KCALMOL = 627.5096080306


def process_input(data, grids):
    """
    process the input
    """
    data_grids_norm = np.zeros((4, len(grids.coord_list), grids.n_rad, grids.n_ang))
    for oxyz in range(4):
        if oxyz == 0:
            data_grids_norm[oxyz, :, :, :] = grids.vector_to_matrix(data[oxyz, :])
        else:
            data_grids_norm[oxyz, :, :, :] = grids.vector_to_matrix(
                np.abs(data[oxyz, :])
            )
    return data_grids_norm


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

    def __init__(self, dict_batch, batch_size, dtype, dict_const=None):
        self.dict_batch = dict_batch
        self.dict_const = dict_const
        self.ids = list(dict_batch["input"].keys())
        self.batch_size = batch_size
        if dtype == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        dict_out = {}
        for key, val in self.dict_batch.items():
            dict_out[key] = val[idx]
        return dict_out

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
            for key, val in batch.items():
                batch_gpu[key] = process(val, self.dtype)
            if self.dict_const is not None:
                for key, val in self.dict_const.items():
                    batch_gpu[key] = torch.tensor(val, dtype=self.dtype).to(
                        device="cuda"
                    )
            dataloader_gpu.append(batch_gpu)
            if self.dict_const is not None:
                if len(dataloader_gpu) > 1:
                    raise ValueError("Only one batch is allowed.")
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
        train_atom_list,
        input_size,
        output_size,
        basis,
        batch_size,
        device,
        dtype,
    ):
        self.molecular_list = molecular_list
        self.extend_atom = extend_atom
        self.extend_xyz = extend_xyz
        self.distance_list = distance_list
        self.train_atom_list = train_atom_list
        self.input_size = input_size
        self.output_size = output_size
        self.basis = basis
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.distance_l = gen_logger(self.distance_list)
        self.data = {}
        self.data_gpu = {}
        self.ene = {}
        self.shape = {}

        self.name_list = []
        self.rng = np.random.default_rng()

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

            if "openshell" in name:
                for i_spin in range(2):
                    name_ = f"{name}_{i_spin}"
                    if not (Path(f"{DATA_PATH}") / f"data_{name_}.npz").exists():
                        print(f"No file: {name_}:>40")
                        continue
                    self.load_data(f"{name_}", Mol[name_mol])
            else:
                if not (Path(f"{DATA_PATH}") / f"data_{name}.npz").exists():
                    print(f"No file: {name:>40}")
                    continue
                self.load_data(name, Mol[name_mol])

    def load_data(self, name, mol):
        """
        Load the data.
        """
        data = np.load(Path(f"{DATA_PATH}") / f"data_{name}.npz")

        if "error_ene_scf" in data.files:
            tot_correct_energy = data["error_ene_scf"]
            print(f"tot_correct_energy: {tot_correct_energy}")
            if np.abs(tot_correct_energy) * AU2KCALMOL > 25:
                return
        else:
            tot_correct_energy = 0

        weight = data["weights"]
        input_mat = data["rho_inv_4_norm"]
        middle_mat = data["vxc1_lda"]
        output_mat = data["exc1_tr_lda"]

        self.name_list.append(name)
        self.shape[name] = output_mat.shape
        input_ = {}
        middle_ = {}
        weight_ = {}
        output_ = {}

        for i_atom in range(output_mat.shape[0]):
            if mol[i_atom][0] not in self.train_atom_list:
                continue

            if self.input_size == 1:
                input_[i_atom] = input_mat[[0], i_atom, :, :]
            elif self.input_size == 4:
                input_[i_atom] = input_mat[:, i_atom, :, :]
            else:
                raise ValueError("input_size should be 1 or 4.")

            weight_[i_atom] = weight[[i_atom], :, :]

            if self.output_size == 1:
                middle_[i_atom] = middle_mat[[i_atom], :, :]
                output_[i_atom] = output_mat[[i_atom], :, :]
            elif self.output_size == 2:
                middle_[i_atom] = middle_mat[[i_atom], :, :]
                output_[i_atom] = np.append(
                    middle_mat[[i_atom], :, :],
                    output_mat[[i_atom], :, :],
                    axis=0,
                )
            elif self.output_size == -1:
                middle_[i_atom] = middle_mat[[i_atom], :, :]
                output_[i_atom] = output_mat[[i_atom], :, :]
            elif self.output_size == -2:
                middle_[i_atom] = middle_mat[[i_atom], :, :]
                output_[i_atom] = output_mat[[i_atom], :, :]
            else:
                raise ValueError("output_size should be -1, 1 or 2.")

            print(
                f"Load {name:>30}, key_: {i_atom:>3}\n"
                f"mean input: {np.mean(input_[i_atom]):>7.4e}, "
                f"max input: {np.max(input_[i_atom]):>7.4e}, "
                f"var input: {np.var(input_[i_atom]):>7.4e}\n"
                f"mean middle: {np.mean(middle_[i_atom]):>7.4e}, "
                f"max middle: {np.max(np.abs(middle_[i_atom])):>7.4e}, "
                f"var middle: {np.var(middle_[i_atom]):>7.4e}\n"
                f"mean output: {np.mean(output_[i_atom]):>7.4e}, "
                f"max output: {np.max(np.abs(output_[i_atom])):>7.4e} "
                f"var output: {np.var(output_[i_atom]):>7.4e}\n"
            )

        for key, val in input_.items():
            tot_correct_energy += np.sum(val * output_[key] * weight_[key])

        self.data_gpu[name] = BasicDataset(
            {
                "input": input_,
                "middle": middle_,
                "output": output_,
                "weight": weight_,
            },
            self.batch_size,
            self.dtype,
            dict_const={"tot_correct_energy": tot_correct_energy},
        ).load_to_gpu()
