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
from pyscf.scf.jk import get_jk
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

    # 1. Test loop
    name_list = []
    time_cc_l, time_dft_l = [], []
    error_scf_rho_r_l, error_dft_rho_r_l = [], []
    dipole_x_diff_scf_l, dipole_y_diff_scf_l, dipole_z_diff_scf_l = [], [], []
    dipole_x_diff_dft_l, dipole_y_diff_dft_l, dipole_z_diff_dft_l = [], [], []
    error_scf_ene_l, error_dft_ene_l = [], []
    abs_scf_ene_l, abs_dft_ene_l, abs_cc_ene_l = [], [], []

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
        # 1.0 Prepare

        molecular = copy.deepcopy(Mol[name_mol])

        index_dict = {}
        for i_atom in ["H", "C"]:
            index_dict[i_atom] = []

        for i in range(len(molecular)):
            index_dict[molecular[i][0]].append(i)

        print(f"Generate {name_mol}_{distance:.4f}", flush=True)
        print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}", flush=True)

        if abs(distance) < 1e-3:
            if (extend_atom != 0) or extend_xyz != 1:
                print(f"Skip: {name:>40}")
                continue

        if extend_atom >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            continue

        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        name_list.append(name)

        molecular[extend_atom][extend_xyz] += distance
        if (DATA_PATH / f"data_{name}.npz").exists():
            data_real = np.load(DATA_PATH / f"data_{name}.npz")
        else:
            print(f"No file: {name:>40}")
            break

        dft2cc = CC_DFT_DATA(
            molecular,
            name=name,
            basis=args.basis,
            if_basis_str=args.if_basis_str,
        )
        dft2cc.test_mol()
        nocc = dft2cc.mol.nelec[0]
        mdft = pyscf.scf.RKS(dft2cc.mol)

        # 1.1 SCF loop to get the density matrix
        time_start = timer()

        # dm1_scf = dft2cc.dm1_dft
        dm1_scf = data_real["dm_inv"] * 2
        oe_fock = oe.contract_expression(
            "p,p,pa,pb->ab",
            np.shape(dft2cc.ao_0[:, 0]),
            np.shape(dft2cc.ao_0[:, 0]),
            dft2cc.ao_0,
            dft2cc.ao_0,
            constants=[2, 3],
            optimize="optimal",
        )

        if args.precision == "float32":
            max_error_scf = 1e-4
        else:
            max_error_scf = 1e-8

        diis = DIIS(dft2cc.mol.nao, n=8)

        for i in range(100):
            middle_mat = data_real["vxc"]
            vxc_scf = dft2cc.grids.matrix_to_vector(middle_mat)

            # inv_r_3 = pyscf.dft.numint.eval_rho(
            #     dft2cc.mol, dft2cc.ao_1, dm1_scf, xctype="GGA"
            # )
            # exc_b3lyp = pyscf.dft.libxc.eval_xc("b3lyp", inv_r_3)[0]
            # vxc_scf += exc_b3lyp

            vxc_mat = oe_fock(
                vxc_scf,
                dft2cc.grids.weights,
            )
            vj_scf = mdft.get_j(dft2cc.mol, dm1_scf)
            mat_fock = dft2cc.h1e + vj_scf + vxc_mat

            diis.add(
                mat_fock,
                dft2cc.mat_s @ dm1_scf @ mat_fock - mat_fock @ dm1_scf @ dft2cc.mat_s,
            )
            mat_fock = diis.hybrid()

            _, mo_scf = np.linalg.eigh(dft2cc.mat_hs @ mat_fock @ dft2cc.mat_hs)
            mo_scf = dft2cc.mat_hs @ mo_scf

            dm1_scf_old = dm1_scf.copy()
            dm1_scf = 2 * mo_scf[:, :nocc] @ mo_scf[:, :nocc].T
            error_dm1 = np.linalg.norm(dm1_scf - dm1_scf_old)

            print(
                f"step:{i:<8}",
                f"dm: {error_dm1::<10.5e}",
            )
            if (i > 0) and (error_dm1 < max_error_scf):
                dm1_scf = dm1_scf_old.copy()
                break

        # 2 check
        # 2.1 check the difference of density (on grids) and dipole
        print(
            f"cc: {dft2cc.time_cc:.2f}s, aidft: {(timer() - time_start):.2f}s",
            flush=True,
        )
        time_cc_l.append(dft2cc.time_cc)
        time_dft_l.append(timer() - time_start)

        del oe_fock
        gc.collect()
        torch.cuda.empty_cache()

        scf_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0,
            dm1_scf,
        )
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
        error_scf_rho_r = np.sum(np.abs(scf_rho_r - cc_rho_r) * dft2cc.grids.weights)
        error_dft_rho_r = np.sum(np.abs(dft_rho_r - cc_rho_r) * dft2cc.grids.weights)
        print(
            f"error_scf_rho_r: {error_scf_rho_r:.2e}, error_dft_rho_r: {error_dft_rho_r:.2e}",
            flush=True,
        )
        error_scf_rho_r_l.append(error_scf_rho_r)
        error_dft_rho_r_l.append(error_dft_rho_r)
        vj_scf = mdft.get_j(dft2cc.mol, dm1_scf)

        dipole_x_core = 0
        for i_atom in range(dft2cc.mol.natm):
            dipole_x_core += (
                dft2cc.mol.atom_charges()[i_atom] * dft2cc.mol.atom_coords()[i_atom][0]
            )
        dipole_x = dipole_x_core - np.sum(
            cc_rho_r * dft2cc.grids.coords[:, 0] * dft2cc.grids.weights
        )
        dipole_x_scf = dipole_x_core - np.sum(
            scf_rho_r * dft2cc.grids.coords[:, 0] * dft2cc.grids.weights
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
        dipole_y_scf = dipole_y_core - np.sum(
            scf_rho_r * dft2cc.grids.coords[:, 1] * dft2cc.grids.weights
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
        dipole_z_scf = dipole_z_core - np.sum(
            scf_rho_r * dft2cc.grids.coords[:, 2] * dft2cc.grids.weights
        )
        dipole_z_dft = dipole_z_core - np.sum(
            dft_rho_r * dft2cc.grids.coords[:, 2] * dft2cc.grids.weights
        )

        print(
            f"dipole_x, cc: {dipole_x:.4f}, scf {dipole_x_scf:.4f}, dft {dipole_x_dft:.4f}"
        )
        print(
            f"dipole_y, cc: {dipole_y:.4f}, scf {dipole_y_scf:.4f}, dft {dipole_y_dft:.4f}"
        )
        print(
            f"dipole_z, cc: {dipole_z:.4f}, scf {dipole_z_scf:.4f}, dft {dipole_z_dft:.4f}",
            flush=True,
        )
        dipole_x_diff_scf_l.append(dipole_x_scf - dipole_x)
        dipole_y_diff_scf_l.append(dipole_y_scf - dipole_y)
        dipole_z_diff_scf_l.append(dipole_z_scf - dipole_z)
        dipole_x_diff_dft_l.append(dipole_x_dft - dipole_x)
        dipole_y_diff_dft_l.append(dipole_y_dft - dipole_y)
        dipole_z_diff_dft_l.append(dipole_z_dft - dipole_z)

        # 2.3 check the difference of energy (total)

        inv_r_3 = pyscf.dft.numint.eval_rho(
            dft2cc.mol, dft2cc.ao_1, dm1_scf, xctype="GGA"
        )
        output_mat = data_real["exc1_tr_b3lyp"]

        output_mat_exc = output_mat * dft2cc.grids.vector_to_matrix(
            scf_rho_r * dft2cc.grids.weights
        )

        inv_r_3 = pyscf.dft.numint.eval_rho(
            dft2cc.mol, dft2cc.ao_1, dm1_scf, xctype="GGA"
        )
        exc_b3lyp = pyscf.dft.libxc.eval_xc("b3lyp", inv_r_3)[0]

        b3lyp_ene = np.sum(exc_b3lyp * scf_rho_r * dft2cc.grids.weights)

        print(AU2KCALMOL * b3lyp_ene)
        print(
            AU2KCALMOL
            * np.sum(
                (data_real["exc1_tr"] - data_real["exc1_tr_b3lyp"])
                * data_real["rho_inv"]
                * data_real["weights"]
            ),
            AU2KCALMOL
            * (
                oe.contract("ij,ji->", dft2cc.h1e, dm1_scf)
                + 0.5 * oe.contract("ij,ji->", vj_scf, dm1_scf)
                + dft2cc.mol.energy_nuc()
                + np.sum(
                    data_real["exc1_tr_b3lyp"]
                    * data_real["rho_inv"]
                    * data_real["weights"]
                )
                + b3lyp_ene
                - dft2cc.e_cc
            ),
        )

        ene_scf = (
            oe.contract("ij,ji->", dft2cc.h1e, dm1_scf)
            + 0.5 * oe.contract("ij,ji->", vj_scf, dm1_scf)
            + dft2cc.mol.energy_nuc()
            + np.sum(output_mat_exc)
            + b3lyp_ene
        )
        error_ene_scf = AU2KCALMOL * (ene_scf - dft2cc.e_cc)
        error_ene_dft = AU2KCALMOL * (dft2cc.e_dft - dft2cc.e_cc)
        print(
            f"error_scf_ene: {error_ene_scf:.2e}, error_dft_ene: {error_ene_dft:.2e}",
            flush=True,
        )

        error_scf_ene_l.append(error_ene_scf)
        error_dft_ene_l.append(error_ene_dft)
        abs_scf_ene_l.append(AU2KCALMOL * ene_scf)
        abs_dft_ene_l.append(AU2KCALMOL * dft2cc.e_dft)
        abs_cc_ene_l.append(AU2KCALMOL * dft2cc.e_cc)

        df = pd.DataFrame(
            {
                "name": name_list,
                "error_scf_ene": error_scf_ene_l,
                "error_dft_ene": error_dft_ene_l,
                "abs_scf_ene_l": abs_scf_ene_l,
                "abs_dft_ene_l": abs_dft_ene_l,
                "abs_cc_ene_l": abs_cc_ene_l,
                "error_scf_rho_r": error_scf_rho_r_l,
                "error_dft_rho_r": error_dft_rho_r_l,
                "dipole_x_diff_scf": dipole_x_diff_scf_l,
                "dipole_y_diff_scf": dipole_y_diff_scf_l,
                "dipole_z_diff_scf": dipole_z_diff_scf_l,
                "dipole_x_diff_dft": dipole_x_diff_dft_l,
                "dipole_y_diff_dft": dipole_y_diff_dft_l,
                "dipole_z_diff_dft": dipole_z_diff_dft_l,
                "time_cc": time_cc_l,
                "time_dft": time_dft_l,
            }
        )
        df.to_csv(
            Path(
                f"{MAIN_PATH}/validate/ccdft_{args.load}_{args.hidden_size}_{args.num_layers}_{args.residual}"
            ),
            index=False,
        )
