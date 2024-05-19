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
        self.dir_weight = data_path / "weight/"
        self.dir_output = data_path / "output/"
        self.dir_input = data_path / "input"

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
            if not (self.dir_output / f"output_cc_dft_diff_{name}.npy").exists():
                print(
                    f"\rNo file: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                    end="",
                )
                continue

            self.load_data(name_mol, name)

    def load_data(self, name_mol, name):
        """
        Load the data.
        """
        e_cc = np.load(self.dir_weight / f"e_ccsd_{name}.npy")
        e_dft = np.load(self.dir_weight / f"e_dft_{name}.npy")
        energy_nuc = np.load(self.dir_weight / f"energy_nuc_{name}.npy")
        aoslice_by_atom = np.load(self.dir_weight / f"aoslice_by_atom_{name}.npy")

        input_mat = np.load(self.dir_input / f"input_dft_{name}.npy")
        middle_mat = np.load(self.dir_input / f"input_cc_{name}.npy")
        output_mat = np.load(self.dir_output / f"output_delta_exc_cc_{name}.npy")
        # output_mat = np.load(self.dir_output / f"output_cc_dft_diff_{name}.npy")

        self.data[name] = {
            "e_cc": e_cc,
            "e_dft": e_dft,
            "energy_nuc": energy_nuc,
        }

        molecular = Mol[name_mol]
        natom = len(molecular)

        for i, j in product(range(natom), range(natom)):
            slice_i = slice(aoslice_by_atom[i][0], aoslice_by_atom[i][1])
            slice_j = slice(aoslice_by_atom[j][0], aoslice_by_atom[j][1])
            slice_ = (slice_i, slice_j)
            if molecular[i][0] != molecular[j][0]:
                key = f"{molecular[i][0]}-{molecular[j][0]}"
            else:
                if i == j:
                    key = f"{molecular[i][0]}-{molecular[j][0]}-D"
                else:
                    key = f"{molecular[i][0]}-{molecular[j][0]}-O"

            self.input[key][f"{name}_{i}_{j}"] = input_mat[slice_].flatten()
            self.middle[key][f"{name}_{i}_{j}"] = (
                middle_mat[slice_].flatten() - input_mat[slice_].flatten()
            )
            self.output[key][f"{name}_{i}_{j}"] = output_mat[slice_].flatten() * 1000

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

            (
                ene_loss_i,
                ene_loss_i_1,
                ene_loss_i_2,
                rho_loss_i,
                dip_x_loss_i,
                dip_y_loss_i,
                dip_z_loss_i,
                name,
            ) = self.check_iter(
                name_mol,
                extend_atom,
                extend_xyz,
                distance,
                model_list,
            )

            ene_loss.append(ene_loss_i)
            ene_loss_1.append(ene_loss_i_1)
            ene_loss_2.append(ene_loss_i_2)
            rho_loss.append(rho_loss_i)
            dip_x_loss.append(dip_x_loss_i)
            dip_y_loss.append(dip_y_loss_i)
            dip_z_loss.append(dip_z_loss_i)
            name_train.append(name)

        return (
            ene_loss,
            ene_loss_1,
            ene_loss_2,
            rho_loss,
            dip_x_loss,
            dip_y_loss,
            dip_z_loss,
            name_train,
        )

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
        exc_real = 0
        delta_exc = 0

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
            output_real = self.output[key][f"{name}_{i}_{j}"]

            dm1_middle_real[
                dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
            ] = (input_mat + middle_real).reshape(
                NAO[molecular[i][0]], NAO[molecular[j][0]]
            )
            exc_real += np.sum(output_real)

            if not (model_list is None):
                input_mat = (
                    torch.as_tensor(input_mat.copy())
                    .to(torch.float64)
                    .contiguous()
                    .to(device=self.device)
                )
                model_list[key + "1"].eval()
                model_list[key + "2"].eval()
                with torch.no_grad():
                    middle_mat = model_list[key + "1"](input_mat)
                    middle_mat += input_mat
                    output_mat = model_list[key + "2"](middle_mat)

                middle_mat = middle_mat.detach().cpu().numpy()
                output_mat = output_mat.detach().cpu().numpy()
            else:
                middle_mat = middle_real.copy()
                middle_mat += input_mat
                output_mat = output_real.copy()

            dm1_middle[dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]] = (
                middle_mat.reshape(NAO[molecular[i][0]], NAO[molecular[j][0]])
            )
            exc += np.sum(output_mat)
            delta_exc += np.sum(np.abs(output_mat - output_real))

        mdft = pyscf.scf.RKS(dft2cc.mol)
        mdft.xc = "b3lyp"
        mdft.grids.kernel()
        coords = mdft.grids.coords
        weights = mdft.grids.weights
        ao_value = dft.numint.eval_ao(dft2cc.mol, coords, deriv=1)

        rho = dft.numint.eval_rho(dft2cc.mol, ao_value, dm1_middle, xctype="GGA")
        exc_cc_grids = dft.libxc.eval_xc("b3lyp", rho)[0]

        h1e = dft2cc.mol.intor("int1e_nuc") + dft2cc.mol.intor("int1e_kin")
        eri = dft2cc.mol.intor("int2e")

        exc_cc_grids = dft.libxc.eval_xc("b3lyp", rho)[0]
        ek_mat_cc = np.einsum("pqrs,pr,qs->qs", eri, dm1_middle, dm1_middle)
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

        rho_real = dft.numint.eval_rho(
            dft2cc.mol, ao_value, dm1_middle_real, xctype="GGA"
        )
        exc_cc_grids_real = dft.libxc.eval_xc("b3lyp", rho_real)[0]
        ek_mat_cc_real = np.einsum(
            "pqrs,pr,qs->qs", eri, dm1_middle_real, dm1_middle_real
        )
        exc_cc_real = (
            np.einsum("i,i,i->", exc_cc_grids_real, rho_real[0], weights)
            - np.sum(ek_mat_cc_real) * 0.05
        )
        e_dft_real = (
            exc_cc_real
            + np.einsum("pqrs,pq,rs", eri, dm1_middle_real, dm1_middle_real) / 2
            + np.sum(h1e * dm1_middle_real)
            + dft2cc.mol.energy_nuc()
        )

        if model_list is None:
            ene_loss_i = exc + 1000 * (e_dft - self.data[name]["e_cc"])
            ene_loss_i_1 = exc - exc_real
            ene_loss_i_2 = 1000 * (e_dft - e_dft_real)
            if ene_loss_i > 1e-3:
                print("")
                print(f"name: {name}, ene_loss_i: {ene_loss_i:7.4f}")

            rho_loss_i = 0
            dip_x_loss_i = 0
            dip_y_loss_i = 0
            dip_z_loss_i = 0
        else:
            ene_loss_i = exc + 1000 * (e_dft - self.data[name]["e_cc"])
            ene_loss_i_1 = exc - exc_real
            ene_loss_i_2 = 1000 * (e_dft - e_dft_real)

            rho_real = dft.numint.eval_rho(dft2cc.mol, ao_value[0], dm1_middle_real)
            rho_loss_i = np.einsum("i,i->", np.abs(rho[0] - rho_real), weights)
            dip_x_loss_i = np.sum(
                np.einsum("i,i,i->i", rho[0] - rho_real, coords[:, 0], weights)
            )
            dip_y_loss_i = np.sum(
                np.einsum("i,i,i->i", rho[0] - rho_real, coords[:, 1], weights)
            )
            dip_z_loss_i = np.sum(
                np.einsum("i,i,i->i", rho[0] - rho_real, coords[:, 2], weights)
            )

        print(
            f"    ene_loss: {ene_loss_i:7.4f}, {ene_loss_i_1:7.4f}, {ene_loss_i_2:7.4f}, rho_loss: {rho_loss_i:7.4f},  delta_exc: {delta_exc:7.4f}.",
            end="",
        )

        return (
            ene_loss_i,
            ene_loss_i_1,
            ene_loss_i_2,
            rho_loss_i,
            dip_x_loss_i,
            dip_y_loss_i,
            dip_z_loss_i,
            name,
        )

    def check_dft(self, model_list=None, if_equilibrium=True):
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

        if not (np.abs(self.distance_l) < 1e-4).any():
            for name_mol in self.molecular_list:
                name = f"{name_mol}_{0}_{1}_{0:.4f}"
                self.load_data(name_mol, name)

                (
                    ene_loss_i,
                    ene_loss_i_1,
                    ene_loss_i_2,
                    rho_loss_i,
                    dip_x_loss_i,
                    dip_y_loss_i,
                    dip_z_loss_i,
                    name,
                ) = self.check_dft_iter(name_mol, 0, 1, 0, model_list)

                ene_loss.append(ene_loss_i)
                ene_loss_1.append(ene_loss_i_1)
                ene_loss_2.append(ene_loss_i_2)
                rho_loss.append(rho_loss_i)
                dip_x_loss.append(dip_x_loss_i)
                dip_y_loss.append(dip_y_loss_i)
                dip_z_loss.append(dip_z_loss_i)
                name_train.append(name)

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

            (
                ene_loss_i,
                ene_loss_i_1,
                ene_loss_i_2,
                rho_loss_i,
                dip_x_loss_i,
                dip_y_loss_i,
                dip_z_loss_i,
                name,
            ) = self.check_dft_iter(
                name_mol,
                extend_atom,
                extend_xyz,
                distance,
                model_list,
            )

            ene_loss.append(ene_loss_i)
            ene_loss_1.append(ene_loss_i_1)
            ene_loss_2.append(ene_loss_i_2)
            rho_loss.append(rho_loss_i)
            dip_x_loss.append(dip_x_loss_i)
            dip_y_loss.append(dip_y_loss_i)
            dip_z_loss.append(dip_z_loss_i)
            name_train.append(name)

        return (
            ene_loss,
            ene_loss_1,
            ene_loss_2,
            rho_loss,
            dip_x_loss,
            dip_y_loss,
            dip_z_loss,
            name_train,
        )

    def check_dft_iter(
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

        for i, j in product(range(dft2cc.mol.natm), range(dft2cc.mol.natm)):
            if molecular[i][0] != molecular[j][0]:
                key = f"{molecular[i][0]}-{molecular[j][0]}"
            else:
                if i == j:
                    key = f"{molecular[i][0]}-{molecular[j][0]}-D"
                else:
                    key = f"{molecular[i][0]}-{molecular[j][0]}-O"

            input_mat = self.input[key][f"{name}_{i}_{j}"]
            middle_real = (
                self.middle[key][f"{name}_{i}_{j}"] + self.input[key][f"{name}_{i}_{j}"]
            )
            dm1_middle[dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]] = (
                input_mat.reshape(NAO[molecular[i][0]], NAO[molecular[j][0]])
            )
            dm1_middle_real[
                dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
            ] = middle_real.reshape(NAO[molecular[i][0]], NAO[molecular[j][0]])

        mdft = pyscf.scf.RKS(dft2cc.mol)
        mdft.xc = "b3lyp"
        mdft.grids.kernel()
        coords = mdft.grids.coords
        weights = mdft.grids.weights
        ao_value = dft.numint.eval_ao(dft2cc.mol, coords, deriv=1)

        rho_real = dft.numint.eval_rho(dft2cc.mol, ao_value[0], dm1_middle_real)
        rho = dft.numint.eval_rho(dft2cc.mol, ao_value[0], dm1_middle)
        rho_loss_i = np.einsum("i,i->", np.abs(rho - rho_real), weights)
        dip_x_loss_i = np.sum(
            np.einsum("i,i,i->i", rho - rho_real, coords[:, 0], weights)
        )
        dip_y_loss_i = np.sum(
            np.einsum("i,i,i->i", rho - rho_real, coords[:, 1], weights)
        )
        dip_z_loss_i = np.sum(
            np.einsum("i,i,i->i", rho - rho_real, coords[:, 2], weights)
        )
        ene_loss_i = 1000 * (self.data[name]["e_dft"] - self.data[name]["e_cc"])
        print(
            f"    ene_loss: {ene_loss_i:7.4f} rho_loss:  {rho_loss_i:7.4f}",
            end="",
        )
        return (
            ene_loss_i,
            0,
            0,
            rho_loss_i,
            dip_x_loss_i,
            dip_y_loss_i,
            dip_z_loss_i,
            name,
        )
