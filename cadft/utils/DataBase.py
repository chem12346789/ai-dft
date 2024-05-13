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
        keys_l,
        molecular_list,
        device,
    ):
        self.args = args
        self.keys_l = keys_l
        self.molecular_list = molecular_list
        self.device = device

        data_path = Path("data")
        dir_weight = data_path / "weight/"
        dir_output = data_path / "output/"

        self.distance_l = gen_logger(args.distance_list)
        self.data = {}
        self.input = {}
        self.middle = {}
        self.output = {}

        for atom_name in keys_l:
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

            if extend_atom >= len(Mol[name_mol]):
                print(
                    f"\rSkip: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                    end="",
                )
                continue

            name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            if not (dir_weight / f"e_ccsd_{name}.npy").exists():
                print(
                    f"\rNo file: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                    end="",
                )
                continue
            e_cc = np.load(dir_weight / f"e_ccsd_{name}.npy")
            e_dft = np.load(dir_weight / f"e_dft_{name}.npy")
            energy_nuc = np.load(dir_weight / f"energy_nuc_{name}.npy")
            cc_dft_diff = np.load(dir_output / f"output_delta_exc_cc_{name}.npy")

            self.data[name] = {
                "e_cc": e_cc,
                "e_dft": e_dft,
                "energy_nuc": energy_nuc,
            }

            molecular = Mol[name_mol]
            natom = len(molecular)

            for i, j in product(range(natom), range(natom)):
                atom_name = molecular[i][0] + molecular[j][0]
                if molecular[i][0] != molecular[j][0]:
                    key = f"{molecular[i][0]}-{molecular[j][0]}"
                else:
                    if i == j:
                        key = f"{molecular[i][0]}-{molecular[j][0]}-D"
                    else:
                        key = f"{molecular[i][0]}-{molecular[j][0]}-O"

                input_path = data_path / "input"
                input_mat = np.load(
                    input_path / f"input_dft_{name}_{i}_{j}.npy"
                ).flatten()
                middle_mat = np.load(
                    input_path / f"input_cc_{name}_{i}_{j}.npy"
                ).flatten()
                output_mat = cc_dft_diff[i, j] * 1000

                self.input[key][f"{name}_{i}_{j}"] = input_mat
                self.middle[key][f"{name}_{i}_{j}"] = middle_mat
                self.output[key][f"{name}_{i}_{j}"] = output_mat[np.newaxis]

    def check(self, model_list=None, if_equilibrium=True):
        """
        Check the input data, if model_list is not none, check loss of the model.
        """
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
            # skip the equilibrium
            if abs(distance) < 1e-3:
                if (extend_atom != 0) or extend_xyz != 1:
                    continue

            if extend_atom >= len(Mol[name_mol]):
                continue

            if if_equilibrium:
                if abs(distance) > 1e-3:
                    continue

            ene_loss_i, rho_loss_i, name = self.check_iter(
                name_mol,
                extend_atom,
                extend_xyz,
                distance,
                model_list,
            )

            ene_loss.append(ene_loss_i)
            rho_loss.append(rho_loss_i)
            name_train.append(name)

        return ene_loss, rho_loss, name_train

    def check_iter(
        self,
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
        model_list=None,
    ):
        """
        Check the input data, if model_list is not none, check loss of the model.
        """
        name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        print(f"\rCheck {name:>30}", end="")

        molecular = copy.deepcopy(Mol[name_mol])
        molecular[extend_atom][extend_xyz] += distance
        dft2cc = cadft.cc_dft_data.CC_DFT_DATA(
            molecular,
            name=name,
            basis=self.args.basis,
            if_basis_str=self.args.if_basis_str,
        )

        dm1_middle = np.zeros((dft2cc.mol.nao, dft2cc.mol.nao))
        dm1_middle_real = np.zeros((dft2cc.mol.nao, dft2cc.mol.nao))
        exc = 0

        for i, j in product(range(dft2cc.mol.natm), range(dft2cc.mol.natm)):
            if molecular[i][0] != molecular[j][0]:
                key = f"{molecular[i][0]}-{molecular[j][0]}"
            else:
                if i == j:
                    key = f"{molecular[i][0]}-{molecular[j][0]}-D"
                else:
                    key = f"{molecular[i][0]}-{molecular[j][0]}-O"

            input_mat = self.input[key][f"{name}_{i}_{j}"]
            middle_real = self.middle[key][f"{name}_{i}_{j}"]
            dm1_middle_real[
                dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
            ] = middle_real.reshape(NAO[molecular[i][0]], NAO[molecular[j][0]])
            output_real = self.output[key][f"{name}_{i}_{j}"]

            if model_list is None:
                output_mat = output_real.copy()
                dm1_middle[
                    dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                ] = dm1_middle_real[
                    dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                ].copy()
            else:
                if not self.args.only_2:
                    input_mat = (
                        torch.as_tensor(input_mat.copy())
                        .to(torch.float64)
                        .contiguous()
                        .to(device=self.device)
                    )
                    middle_mat = model_list[key + "1"](input_mat)
                    output_mat = model_list[key + "2"](middle_mat)
                    middle_mat = middle_mat.detach().cpu().numpy()
                    output_mat = output_mat.detach().cpu().numpy()
                    dm1_middle[
                        dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                    ] = middle_mat.reshape(NAO[molecular[i][0]], NAO[molecular[j][0]])
                else:
                    dm1_middle[
                        dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                    ] = dm1_middle_real[
                        dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                    ].copy()
                    middle_mat = (
                        torch.as_tensor(middle_real.copy())
                        .to(torch.float64)
                        .contiguous()
                        .to(device=self.device)
                    )
                    output_mat = model_list[key + "2"](middle_mat)
                    output_mat = output_mat.detach().cpu().numpy()

            exc += output_mat[0]

        mdft = pyscf.scf.RKS(dft2cc.mol)
        mdft.xc = "b3lyp"
        mdft.kernel()
        eri = dft2cc.mol.intor("int2e")
        h1e = dft2cc.mol.intor("int1e_nuc") + dft2cc.mol.intor("int1e_kin")
        coords = mdft.grids.coords
        weights = mdft.grids.weights
        ao_value = dft.numint.eval_ao(dft2cc.mol, coords, deriv=1)

        ek_mat_cc = np.einsum("pqrs,pr,qs->qs", eri, dm1_middle, dm1_middle)
        rho = dft.numint.eval_rho(dft2cc.mol, ao_value, dm1_middle, xctype="GGA")
        exc_cc_grids = dft.libxc.eval_xc("b3lyp", rho)[0]
        exc_cc = (
            np.einsum("i,i,i->", exc_cc_grids, rho[0], weights)
            - np.sum(ek_mat_cc) * 0.05
        )
        e_dft = (
            exc_cc
            + np.einsum("pqrs,pq,rs", eri, dm1_middle, dm1_middle) / 2
            + np.sum(h1e * dm1_middle)
            + dft2cc.mol.energy_nuc()
        )

        if model_list is None:
            ene_loss_i = exc + 1000 * (e_dft - self.data[name]["e_cc"])
            if ene_loss_i > 1e-3:
                print("")
                print(f"name: {name}, ene_loss_i: {ene_loss_i:7.4f}")

            rho_loss_i = 0
        else:
            ene_loss_i = exc + 1000 * (e_dft - self.data[name]["e_cc"])

            rho_real = dft.numint.eval_rho(dft2cc.mol, ao_value[0], dm1_middle_real)
            rho_loss_i = np.einsum("i,i->", np.abs(rho[0] - rho_real), weights)

        print(
            f"    ene_loss: {ene_loss_i:7.4f} rho_loss:  {rho_loss_i:7.4f}",
            end="",
        )

        return ene_loss_i, rho_loss_i, name
