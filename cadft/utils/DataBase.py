from pathlib import Path
import copy
from itertools import product

import pyscf
import torch
import numpy as np
import opt_einsum as oe
import h5py

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

        if args.hdf5:
            hdf5file = h5py.File(Path("data") / "file.h5", "r")
        else:
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
            if args.hdf5:
                e_cc = hdf5file[f"weight/e_ccsd_{name}"][()]
                energy_nuc = hdf5file[f"weight/energy_nuc_{name}"][()]
            else:
                if not (dir_weight / f"e_ccsd_{name}.npy").exists():
                    print(
                        f"\rNo file: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
                        end="",
                    )
                    continue
                e_cc = np.load(dir_weight / f"e_ccsd_{name}.npy")
                e_dft = np.load(dir_weight / f"e_dft_{name}.npy")
                energy_nuc = np.load(dir_weight / f"energy_nuc_{name}.npy")
                cc_dft_diff = np.load(dir_output / f"output_cc_dft_diff_{name}.npy")

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

                if args.hdf5:
                    input_mat = hdf5file[f"{atom_name}/input/input_{name}_{i}_{j}"][:]
                    output_mat = hdf5file[
                        f"{atom_name}/output/output_exc_{name}_{i}_{j}"
                    ][:]
                else:
                    input_path = data_path / "input"
                    input_mat = np.load(
                        input_path / f"input_dft_{name}_{i}_{j}.npy"
                    ).flatten()
                    output_mat = cc_dft_diff[i, j] * 1000

                self.input[key][f"{name}_{i}_{j}"] = input_mat
                self.output[key][f"{name}_{i}_{j}"] = output_mat[np.newaxis]

        if args.hdf5:
            hdf5file.close()

    def check(self, model_list=None, if_equilibrium=True):
        """
        Check the input data, if model_list is not none, check loss of the model.
        """
        ene_loss = []
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

            dm1_dft = np.zeros((dft2cc.mol.nao, dft2cc.mol.nao))
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
                output_mat_real = self.output[key][f"{name}_{i}_{j}"]
                dm1_dft[dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]] = (
                    input_mat.reshape(NAO[molecular[i][0]], NAO[molecular[j][0]])
                )

                if model_list is None:
                    output_mat = output_mat_real.copy()
                else:
                    input_mat = (
                        torch.as_tensor(input_mat.copy())
                        .to(torch.float64)
                        .contiguous()
                        .to(device=self.device)
                    )
                    output_mat = model_list[key](input_mat)
                    output_mat = output_mat.detach().cpu().numpy()
                exc += output_mat[0]

            if model_list is None:
                ene_loss_i = exc + 1000 * (
                    self.data[name]["e_dft"] - self.data[name]["e_cc"]
                )
                if ene_loss_i > 1e-3:
                    print("")
                    print(f"name: {name}, ene_loss_i: {ene_loss_i:7.4f}")
            else:
                ene_loss_i = exc + 1000 * (
                    self.data[name]["e_dft"] - self.data[name]["e_cc"]
                )

            print(f"    ene_loss: {ene_loss_i:7.4f}", end="")
            ene_loss.append(ene_loss_i)

        return (
            ene_loss,
            name_train,
        )

    def check_rho(self, model_list=None, if_equilibrium=True):
        """
        Check the input data, if model_list is not none, check loss of the model.
        """
        ene_loss = []
        name_train = []
        rho_loss = []
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
            # skip the equilibrium
            if abs(distance) < 1e-3:
                if (extend_atom != 0) or extend_xyz != 1:
                    continue

            if extend_atom >= len(Mol[name_mol]):
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
                    middle_mat_real = self.middle[atom_name][f"{name}_{i}_{j}"]
                    dm1_cc_real[
                        dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]
                    ] = middle_mat_real.reshape(
                        NAO[molecular[i][0]], NAO[molecular[j][0]]
                    )

                    if model_list is None:
                        output_mat = output_mat_real.copy()
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
                if ene_loss_i > 1e-3:
                    print("")
                    print(f"name: {name}, ene_loss_i: {ene_loss_i:7.4f}")
            else:
                ene_loss_i = 1000 * (
                    exc
                    + np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc) / 2
                    + np.sum(h1e * dm1_cc)
                    + self.data[name]["energy_nuc"]
                    - self.data[name]["e_cc"]
                )

            if model_list is None:
                rho_loss_i = 0
                dipole_x_loss_i = 0
                dipole_y_loss_i = 0
                dipole_z_loss_i = 0
            else:
                mdft = pyscf.scf.RKS(dft2cc.mol)
                mdft.grids.build()
                ao = pyscf.dft.numint.eval_ao(dft2cc.mol, mdft.grids.coords)

                if self.args.noise_print:
                    self.args.noise_print += 1
                    print(
                        np.array2string(
                            np.abs(dm1_cc - dm1_cc_real),
                            formatter={"float_kind": lambda x: f"{x:8.6f}"},
                        )
                    )
                    print(np.mean(np.abs(dm1_cc - dm1_cc_real)))

                    if self.args.noise_print > 10:
                        print("noise_print closed!")
                        self.args.noise_print = 0

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
