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
from cadft.utils.Grids import Grid


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

        data_path = Path("data")
        self.dir_grids = data_path / "grids/"
        self.dir_weight = data_path / "weight/"

        self.distance_l = gen_logger(args.distance_list)
        self.data = {}

        self.input = {}
        self.middle = {}
        self.output = {}
        self.weight = {}

        for atom in atom_list:
            self.input[atom] = {}
            self.middle[atom] = {}
            self.output[atom] = {}
            self.weight[atom] = {}

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

            molecular_list = copy.deepcopy(Mol[name_mol])
            name = f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}"
            if not (self.dir_grids / f"data_{name}.npz").exists():
                print(
                    f"No file: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}"
                )
                continue

            self.load_data(name, molecular_list)

    def load_data(self, name, molecular_list):
        """
        Load the data.
        """
        data = np.load(self.dir_grids / f"data_{name}.npz")
        e_cc = np.load(self.dir_weight / f"e_ccsd_{name}.npy")
        e_dft = np.load(self.dir_weight / f"e_dft_{name}.npy")

        weight = data["weights"]
        input_mat = data["rho_dft"] * weight
        middle_mat = data["rho_cc"] * weight
        output_mat = data["exc_over_dm_cc_grids"] * middle_mat

        self.data[name] = {
            "e_cc": e_cc,
            "e_dft": e_dft,
            "ene_vc": data["ene_vc"],
        }

        for i_atom in range(input_mat.shape[0]):
            atom_name = molecular_list[i_atom][0]
            for i in range(input_mat.shape[2]):
                # if np.linalg.norm(input_mat[i_atom, i, :]) < 1e-10:
                #     continue
                self.input[atom_name][f"{name}_{i_atom}_{i}"] = input_mat[i_atom, :, i]
                self.middle[atom_name][f"{name}_{i_atom}_{i}"] = (
                    middle_mat[i_atom, :, i] - input_mat[i_atom, :, i]
                )
                self.output[atom_name][f"{name}_{i_atom}_{i}"] = output_mat[
                    i_atom, :, i
                ]
                self.weight[atom_name][f"{name}_{i_atom}_{i}"] = weight[i_atom, :, i]

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

        mdft = pyscf.scf.RKS(dft2cc.mol)
        mdft.xc = "b3lyp"
        grids = Grid(dft2cc.mol)
        coords = grids.coords
        weights = grids.weights

        weights_shape = grids.vector_to_matrix(weights).shape
        rho_real = np.zeros(weights_shape)
        exc_real = np.zeros(weights_shape)
        rho_pred = np.zeros(weights_shape)
        exc_pred = np.zeros(weights_shape)
        weight = np.zeros(weights_shape)

        for i_atom in range(weights_shape[0]):
            atom_name = molecular[i_atom][0]
            for i in range(weights_shape[2]):
                input_mat = self.input[atom_name][f"{name}_{i_atom}_{i}"]
                middle_mat_real = self.middle[atom_name][f"{name}_{i_atom}_{i}"]
                middle_mat_real += input_mat
                output_mat_real = self.output[atom_name][f"{name}_{i_atom}_{i}"]

                rho_real[i_atom, :, i] = middle_mat_real
                exc_real[i_atom, :, i] = output_mat_real
                weight[i_atom, :, i] = self.weight[atom_name][f"{name}_{i_atom}_{i}"]

                if not (model_list is None):
                    input_mat = (
                        torch.as_tensor(input_mat.copy())
                        .to(torch.float64)
                        .contiguous()
                        .to(device=self.device)
                    )
                    middle_mat_real = (
                        torch.as_tensor(middle_mat_real.copy())
                        .to(torch.float64)
                        .contiguous()
                        .to(device=self.device)
                    )
                    model_list[atom_name + "1"].eval()
                    model_list[atom_name + "2"].eval()
                    with torch.no_grad():
                        middle_mat = model_list[atom_name + "1"](input_mat)
                        middle_mat += input_mat
                        output_mat = model_list[atom_name + "2"](middle_mat)

                    middle_mat = middle_mat.detach().cpu().numpy()
                    output_mat = output_mat.detach().cpu().numpy()
                    print(output_mat)
                    print(output_mat_real)

                    rho_pred[i_atom, :, i] = middle_mat
                    exc_pred[i_atom, :, i] = output_mat
                else:
                    rho_pred[i_atom, :, i] = middle_mat_real
                    exc_pred[i_atom, :, i] = output_mat_real

        if model_list is None:
            ene_loss_i = np.abs(np.sum(exc_pred) - self.data[name]["ene_vc"])
            ene_loss_i_1 = np.sum(np.abs(exc_pred - exc_real))
            ene_loss_i_2 = 0
            if ene_loss_i > 1e-3:
                print("")
                print(f"name: {name}, ene_loss_i: {ene_loss_i:7.4f}")

            rho_loss_i = 0
            dip_x_loss_i = 0
            dip_y_loss_i = 0
            dip_z_loss_i = 0
        else:
            ene_loss_i = np.abs(np.sum(exc_pred) - self.data[name]["ene_vc"])
            ene_loss_i_1 = np.sum(np.abs(exc_pred - exc_real))
            ene_loss_i_2 = 0

            rho_real = grids.matrix_to_vector(rho_real)
            rho_pred = grids.matrix_to_vector(rho_pred)
            weight = grids.matrix_to_vector(weight)
            rho_loss_i = np.sum(np.abs(rho_pred - rho_real))
            dip_x_loss_i = np.sum((rho_pred - rho_real) * coords[:, 0])
            dip_y_loss_i = np.sum((rho_pred - rho_real) * coords[:, 1])
            dip_z_loss_i = np.sum((rho_pred - rho_real) * coords[:, 2])
        print(
            f"    ene_loss: {ene_loss_i:7.4f}, rho_loss: {rho_loss_i:7.4f}, total rho {np.sum(rho_pred):7.4f}, total rho {np.sum(rho_real):7.4f}.",
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
