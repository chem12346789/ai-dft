from pathlib import Path
import copy
from itertools import product

import torch
import numpy as np
import opt_einsum as oe
import pyscf

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
    ):
        self.args = args
        self.atom_list = atom_list
        self.molecular_list = molecular_list
        self.device = device

        self.distance_l = gen_logger(args.distance_list)
        data_path = Path("data")
        self.data = {}
        self.input = {}
        self.middle = {}
        self.output = {}

        for i_atom in self.atom_list:
            for j_atom in self.atom_list:
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
                        f"\rSkip {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                        end="",
                    )
                    continue
            name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            dir_weight = data_path / "weight/"

            e_cc = np.load(dir_weight / f"e_ccsd_{name}.npy")
            energy_nuc = np.load(dir_weight / f"energy_nuc_{name}.npy")

            molecular = Mol[name_mol]
            natom = len(molecular)

            for i in range(natom):
                for j in range(natom):
                    atom_name = molecular[i][0] + molecular[j][0]
                    input_path = data_path / atom_name / "input"
                    output_path = data_path / atom_name / "output"

                    input_mat = np.load(
                        input_path / f"input_{name}_{i}_{j}.npy"
                    ).flatten()
                    self.input[atom_name][f"{name}_{i}_{j}"] = input_mat

                    middle_mat = np.load(
                        output_path / f"output_dm1_{name}_{i}_{j}.npy"
                    ).flatten()
                    self.middle[atom_name][f"{name}_{i}_{j}"] = middle_mat

                    output_mat = np.load(
                        output_path / f"output_exc_{name}_{i}_{j}.npy"
                    ).sum()
                    self.output[atom_name][f"{name}_{i}_{j}"] = output_mat[np.newaxis]

            self.data[name] = {
                "e_cc": e_cc,
                "energy_nuc": energy_nuc,
            }
        print()

    def check(self, model_list=None, if_equilibrium=True):
        ene_loss = []
        rho_loss = []
        name_train = []
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
            print(f"\rCheck {name:>40}", end="")
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
                    middle_mat_real = self.middle[atom_name][f"{name}_{i}_{j}"]

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
                        dm1_cc_real[
                            dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                        ] = middle_mat_real.reshape(
                            NAO[molecular[i][0]], NAO[molecular[j][0]]
                        )
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
                        middle_mat = middle_mat.detach().cpu().numpy()
                        output_mat = output_mat.detach().cpu().numpy()
                        dm1_cc[
                            dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                        ] = middle_mat.reshape(
                            NAO[molecular[i][0]], NAO[molecular[j][0]]
                        )

                    exc += output_mat[0]

            if model_list is None:
                ene_loss.append(
                    np.abs(
                        1000
                        * (
                            exc
                            + np.einsum("pqrs,pq,rs", eri, dm1_cc_real, dm1_cc_real) / 2
                            + np.sum(h1e * dm1_cc_real)
                            + self.data[name]["energy_nuc"]
                            - self.data[name]["e_cc"]
                        )
                    )
                )
            else:
                ene_loss.append(
                    np.abs(
                        1000
                        * (
                            exc
                            + np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc) / 2
                            + np.sum(h1e * dm1_cc)
                            + self.data[name]["energy_nuc"]
                            - self.data[name]["e_cc"]
                        )
                    )
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

                rho_loss.append(1000 * np.sum(np.abs(dm_cc_r - dm_cc_real_r)))

        print()
        return ene_loss, rho_loss, name_train
