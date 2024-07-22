"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
import copy
import gc
from itertools import product
from pathlib import Path
from timeit import default_timer as timer

import pyscf
import torch
import numpy as np
import pandas as pd
import opt_einsum as oe
from torch.utils.data import DataLoader

from cadft import CC_DFT_DATA, add_args, gen_logger
from cadft.utils import ModelDict
from cadft.utils import Mol
from cadft.utils import MAIN_PATH, DATA_PATH


AU2KCALMOL = 627.5096080306


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

    def __init__(self, input_, weight_, batch_size, dtype):
        self.input = input_
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
                batch_gpu["weight"],
            ) = (
                process(batch["input"], self.dtype),
                process(batch["weight"], self.dtype),
            )
            dataloader_gpu.append(batch_gpu)
        return dataloader_gpu


class DIIS:
    """
    DIIS for the Fock matrix.
    """

    def __init__(self, nao, n=50):
        self.n = n
        self.errors = np.zeros((n, nao, nao))
        self.mat_fock = np.zeros((n, nao, nao))
        self.step = 0

    def add(self, mat_fock, error):
        self.mat_fock = np.roll(self.mat_fock, -1, axis=0)
        self.mat_fock[-1, :, :] = mat_fock
        self.errors = np.roll(self.errors, -1, axis=0)
        self.errors[-1, :, :] = error

    def hybrid(self):
        self.step += 1
        mat = np.zeros((self.n + 1, self.n + 1))
        mat[:-1, :-1] = np.einsum("inm,jnm->ij", self.errors, self.errors)
        mat[-1, :] = -1
        mat[:, -1] = -1
        mat[-1, -1] = 0

        b = np.zeros(self.n + 1)
        b[-1] = -1

        if self.step < self.n:
            c = np.linalg.solve(
                mat[-(self.step + 1) :, -(self.step + 1) :], b[-(self.step + 1) :]
            )
            mat_fock = np.sum(
                c[:-1, np.newaxis, np.newaxis] * self.mat_fock[-self.step :], axis=0
            )
            return mat_fock
        else:
            c = np.linalg.solve(mat, b)
            mat_fock = np.sum(c[:-1, np.newaxis, np.newaxis] * self.mat_fock, axis=0)
            return mat_fock


if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init the model
    modeldict = ModelDict(
        args.load,
        args.input_size,
        args.hidden_size,
        args.output_size,
        args.num_layers,
        args.residual,
        device,
        args.precision,
        if_mkdir=False,
    )
    modeldict.load_model()
    modeldict.eval()

    # 2. Test loop
    name_list = []
    error_dft_rho_r_l = []
    dipole_x_diff_dft_l, dipole_y_diff_dft_l, dipole_z_diff_dft_l = [], [], []
    error_dft_ene_l = []
    abs_dft_ene_l, abs_cc_ene_l = [], []

    distance_l = gen_logger(args.distance_list)
    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        args.name_mol,
        args.extend_atom,
        args.extend_xyz,
        distance_l,
    ):
        # 2.0 Prepare

        molecular = copy.deepcopy(Mol[name_mol])

        print(f"Generate {name_mol}_{distance:.4f}", flush=True)
        print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}", flush=True)

        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        name_list.append(name)

        if abs(distance) < 1e-3:
            if (extend_atom != 0) or extend_xyz != 1:
                print(f"Skip: {name:>40}")
                continue

        if extend_atom >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            continue

        molecular[extend_atom][extend_xyz] += distance
        if (DATA_PATH / f"data_{name}.npz").exists():
            data_real = np.load(DATA_PATH / f"data_{name}.npz")
        else:
            print(f"No file: {name:>40}")
            data_real = None

        dft2cc = CC_DFT_DATA(
            molecular,
            name=name,
            basis=args.basis,
            if_basis_str=args.if_basis_str,
        )
        dft2cc.test_mol()
        nocc = dft2cc.mol.nelec[0]

        cc_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0,
            dft2cc.dm1_cc,
        )
        dft_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0,
            dft2cc.dm1_dft,
        )
        error_dft_rho_r = np.sum(np.abs(dft_rho_r - cc_rho_r) * dft2cc.grids.weights)
        print(
            f"error_dft_rho_r: {error_dft_rho_r:.2e}",
            flush=True,
        )
        error_dft_rho_r_l.append(error_dft_rho_r)

        dipole_x_core = 0
        for i_atom in range(dft2cc.mol.natm):
            dipole_x_core += (
                dft2cc.mol.atom_charges()[i_atom] * dft2cc.mol.atom_coords()[i_atom][0]
            )
        dipole_x = dipole_x_core - np.sum(
            cc_rho_r * dft2cc.grids.coords[:, 0] * dft2cc.grids.weights
        )
        dipole_x_dft = dipole_x_core - np.sum(
            dft_rho_r * dft2cc.grids.coords[:, 0] * dft2cc.grids.weights
        )

        dipole_y_core = 0
        for i_atom in range(dft2cc.mol.natm):
            dipole_y_core += (
                dft2cc.mol.atom_charges()[i_atom] * dft2cc.mol.atom_coords()[i_atom][1]
            )
        dipole_y = dipole_y_core - np.sum(
            cc_rho_r * dft2cc.grids.coords[:, 1] * dft2cc.grids.weights
        )
        dipole_y_dft = dipole_y_core - np.sum(
            dft_rho_r * dft2cc.grids.coords[:, 1] * dft2cc.grids.weights
        )

        dipole_z_core = 0
        for i_atom in range(dft2cc.mol.natm):
            dipole_z_core += (
                dft2cc.mol.atom_charges()[i_atom] * dft2cc.mol.atom_coords()[i_atom][2]
            )
        dipole_z = dipole_z_core - np.sum(
            cc_rho_r * dft2cc.grids.coords[:, 2] * dft2cc.grids.weights
        )
        dipole_z_dft = dipole_z_core - np.sum(
            dft_rho_r * dft2cc.grids.coords[:, 2] * dft2cc.grids.weights
        )

        print(f"dipole_x, cc: {dipole_x:.4f}, dft {dipole_x_dft:.4f}")
        print(f"dipole_y, cc: {dipole_y:.4f}, dft {dipole_y_dft:.4f}")
        print(
            f"dipole_z, cc: {dipole_z:.4f}, dft {dipole_z_dft:.4f}",
            flush=True,
        )
        dipole_x_diff_dft_l.append(dipole_x_dft - dipole_x)
        dipole_y_diff_dft_l.append(dipole_y_dft - dipole_y)
        dipole_z_diff_dft_l.append(dipole_z_dft - dipole_z)

        # 2.3 check the difference of energy (total)

        error_ene_dft = AU2KCALMOL * (dft2cc.e_dft - dft2cc.e_cc)
        print(
            f"error_dft_ene: {error_ene_dft:.2e}",
            flush=True,
        )

        error_dft_ene_l.append(error_ene_dft)
        abs_dft_ene_l.append(AU2KCALMOL * dft2cc.e_dft)
        abs_cc_ene_l.append(AU2KCALMOL * dft2cc.e_cc)

        df = pd.DataFrame(
            {
                "name": name_list,
                "error_dft_ene": error_dft_ene_l,
                "abs_dft_ene_l": abs_dft_ene_l,
                "abs_cc_ene_l": abs_cc_ene_l,
                "error_dft_rho_r": error_dft_rho_r_l,
                "dipole_x_diff_dft": dipole_x_diff_dft_l,
                "dipole_y_diff_dft": dipole_y_diff_dft_l,
                "dipole_z_diff_dft": dipole_z_diff_dft_l,
            }
        )
        df.to_csv(
            Path(
                f"{MAIN_PATH}/validate/ccdft_{args.load}_{args.hidden_size}_{args.num_layers}_{args.residual}"
            ),
            index=False,
        )
