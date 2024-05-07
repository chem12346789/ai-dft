from pathlib import Path
import copy
from itertools import product

import pyscf
import torch
import numpy as np
import opt_einsum as oe

from cadft.utils.logger import gen_logger
from cadft.utils.nao import NAO
from cadft.utils.mol import Mol
import cadft


class DataBase:
    """Documentation for a class."""

    def __init__(
        self,
        args,
        atom_list,
        molecular_list,
        device,
        normalize=False,
    ):
        self.args = args
        self.atom_list = atom_list
        self.molecular_list = molecular_list
        self.device = device
        self.normalize = normalize

        self.distance_l = gen_logger(args.distance_list)
        data_path = Path("data")
        self.data = {}
        self.input = {}
        self.middle = {}
        self.output = {}

        for i_atom, j_atom in product(self.atom_list, self.atom_list):
            atom_name = i_atom + j_atom
            self.input[atom_name] = {}
            self.middle[atom_name] = {}
            self.output[atom_name] = {}

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
                        f"\rSkip: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                        end="",
                    )
                    continue
            name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            dir_weight = data_path / "weight/"
            if not (dir_weight / f"e_ccsd_{name}.npy").exists():
                print(
                    f"\rNo file: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                    end="",
                )
                continue
            e_cc = np.load(dir_weight / f"e_ccsd_{name}.npy")
            energy_nuc = np.load(dir_weight / f"energy_nuc_{name}.npy")
            self.data[name] = {
                "e_cc": e_cc,
                "energy_nuc": energy_nuc,
            }

            molecular = Mol[name_mol]
            natom = len(molecular)

            for i, j in product(range(natom), range(natom)):
                atom_name = molecular[i][0] + molecular[j][0]
                input_path = data_path / atom_name / "input"
                output_path = data_path / atom_name / "output"

                input_mat = np.load(input_path / f"input_{name}_{i}_{j}.npy").flatten()
                self.input[atom_name][f"{name}_{i}_{j}"] = input_mat

                middle_mat = np.load(
                    output_path / f"output_dm1_{name}_{i}_{j}.npy"
                ).flatten()
                if self.normalize:
                    middle_mat = middle_mat / (np.cosh(input_mat) - 0.95)
                self.middle[atom_name][f"{name}_{i}_{j}"] = middle_mat

                output_mat = np.load(
                    output_path / f"output_exc_{name}_{i}_{j}.npy"
                ).sum()
                self.output[atom_name][f"{name}_{i}_{j}"] = output_mat[np.newaxis]

    def check(self, model_list=None, if_equilibrium=True):
        """
        Check the input data, if model_list is not none, check loss of the model.
        """
        ene_loss = []
        rho_loss = []
        name_train = []
        dipole_x_loss = []
        dipole_y_loss = []
        dipole_z_loss = []

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
                    continue
            if if_equilibrium:
                if abs(distance) > 1e-3:
                    continue
            name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            print(f"\rCheck {name:>30}", end="")
            name_train.append(name)

            molecular = copy.deepcopy(Mol[name_mol])
            molecular[extend_atom][extend_xyz] += distance
            dft2cc = cadft.cc_dft_data.CC_DFT_DATA(
                molecular,
                name=name,
                basis=self.args.basis,
                if_basis_str=self.args.if_basis_str,
            )

            eri = dft2cc.mol.intor("int2e")
            h1e = dft2cc.mol.intor("int1e_nuc") + dft2cc.mol.intor("int1e_kin")

            dm1_cc = np.zeros((dft2cc.mol.nao, dft2cc.mol.nao))
            dm1_cc_real = np.zeros((dft2cc.mol.nao, dft2cc.mol.nao))
            exc = 0

            for i in range(dft2cc.mol.natm):
                for j in range(dft2cc.mol.natm):
                    atom_name = molecular[i][0] + molecular[j][0]
                    input_mat = self.input[atom_name][f"{name}_{i}_{j}"]
                    output_mat_real = self.output[atom_name][f"{name}_{i}_{j}"]
                    middle_mat_real = self.middle[atom_name][f"{name}_{i}_{j}"] * (
                        np.exp(-np.abs(input_mat)) - 0.9999
                    )
                    dm1_cc_real[
                        dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                    ] = middle_mat_real.reshape(
                        NAO[molecular[i][0]], NAO[molecular[j][0]]
                    )
                    # self.weight_h1e[atom_name][f"{name}_{i}_{j}"] = h1e[
                    #     dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                    # ].flatten()
                    # self.weight_eri[atom_name][f"{name}_{i}_{j}"] = eri[
                    #     dft2cc.atom_info["slice"][i],
                    #     dft2cc.atom_info["slice"][j],
                    #     dft2cc.atom_info["slice"][i],
                    #     dft2cc.atom_info["slice"][j],
                    # ].reshape(
                    #     NAO[molecular[i][0]] * NAO[molecular[j][0]],
                    #     NAO[molecular[i][0]] * NAO[molecular[j][0]],
                    # )

                    if model_list is None:
                        output_mat = output_mat_real.copy()
                        # exc += (
                        #     self.weight_h1e[atom_name][f"{name}_{i}_{j}"]
                        #     @ middle_mat_real
                        # )
                    else:
                        input_mat = (
                            torch.as_tensor(input_mat.copy())
                            .to(torch.float64)
                            .contiguous()
                            .to(device=self.device)
                        )
                        middle_mat = model_list[atom_name + "1"](input_mat)
                        output_mat = model_list[atom_name + "2"](middle_mat)
                        input_mat = input_mat.detach().cpu().numpy()
                        middle_mat = middle_mat.detach().cpu().numpy()
                        output_mat = output_mat.detach().cpu().numpy()
                        if self.normalize:
                            middle_mat = middle_mat * (np.cosh(input_mat) - 0.95)
                        dm1_cc[
                            dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                        ] = middle_mat.reshape(
                            NAO[molecular[i][0]], NAO[molecular[j][0]]
                        )

                    exc += output_mat[0]

            if model_list is None:
                ene_loss_i = 1000 * (
                    exc
                    + np.einsum("pqrs,pq,rs", eri, dm1_cc_real, dm1_cc_real) / 2
                    + np.sum(h1e * dm1_cc_real)
                    + self.data[name]["energy_nuc"]
                    - self.data[name]["e_cc"]
                )
            else:
                ene_loss_i = 1000 * (
                    exc
                    + np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc) / 2
                    + np.sum(h1e * dm1_cc)
                    + self.data[name]["energy_nuc"]
                    - self.data[name]["e_cc"]
                )

            if model_list is None:
                rho_loss.append(0)
            else:
                mdft = pyscf.scf.RKS(dft2cc.mol)
                mdft.grids.build()
                ao = pyscf.dft.numint.eval_ao(dft2cc.mol, mdft.grids.coords)

                dm_cc_r = oe.contract(
                    "uv,gu,gv,g->g",
                    dm1_cc,
                    ao,
                    ao,
                    mdft.grids.weights,
                    optimize="auto",
                )
                dm_cc_real_r = oe.contract(
                    "uv,gu,gv,g->g",
                    dm1_cc_real,
                    ao,
                    ao,
                    mdft.grids.weights,
                    optimize="auto",
                )

                rho_loss_i = 1000 * np.sum(np.abs(dm_cc_r - dm_cc_real_r))
                dipole_x_loss_i = 1000 * np.sum(
                    mdft.grids.coords[:, 0] * dm_cc_r
                    - mdft.grids.coords[:, 0] * dm_cc_real_r
                )
                dipole_y_loss_i = 1000 * np.sum(
                    mdft.grids.coords[:, 1] * dm_cc_r
                    - mdft.grids.coords[:, 1] * dm_cc_real_r
                )
                dipole_z_loss_i = 1000 * np.sum(
                    mdft.grids.coords[:, 2] * dm_cc_r
                    - mdft.grids.coords[:, 2] * dm_cc_real_r
                )
            print(
                f"    ene_loss: {ene_loss_i:7.4f}, rho_loss: {rho_loss_i:7.4f}", end=""
            )
            ene_loss.append(ene_loss_i)
            rho_loss.append(rho_loss_i)
            dipole_x_loss.append(dipole_x_loss_i)
            dipole_y_loss.append(dipole_y_loss_i)
            dipole_z_loss.append(dipole_z_loss_i)

        return (
            ene_loss,
            rho_loss,
            dipole_x_loss,
            dipole_y_loss,
            dipole_z_loss,
            name_train,
        )
