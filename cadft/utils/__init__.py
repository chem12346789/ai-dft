import pandas as pd
import copy
import numpy as np
import pyscf

from cadft.utils.basis import gen_basis
from cadft.utils.rotate import rotate
from cadft.utils.parser import add_args
from cadft.utils.mrks import mrks_append
from cadft.utils.umrks import umrks_append
from cadft.utils.mrks_diis import mrks_diis
from cadft.utils.umrks_diis import umrks_diis
from cadft.utils.gmrks_diis import gmrks_diis
from cadft.utils.DataBase import gen_logger, process_input

from cadft.utils.DataBase import DataBase
from cadft.utils.model.fc_net import FCNet
from cadft.utils.model.transformer import Transformer
from cadft.utils.Grids import Grid
from cadft.utils.ModelDict import ModelDict
from cadft.utils.ModelDict_xy import ModelDict_xy
from cadft.utils.ModelDict_xy1 import ModelDict_xy1
from cadft.utils.diis import DIIS

from cadft.utils.mol import Mol
from cadft.utils.env_var import MAIN_PATH, DATA_PATH, DATA_SAVE_PATH, DATA_CC_PATH

ORIENTATION_NUMBER_DICT = {"x": 0, "y": 1, "z": 2}


def save_csv_loss(
    name_list,
    path,
    dict_: dict,
):
    """
    save the loss to a csv file
    """
    dict_empty = {}
    dict_["name"] = name_list
    for key, val in dict_.items():
        dict_empty[key] = val
    df = pd.DataFrame(dict_empty)
    df.to_csv(path, index=False)


def extend(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    basis: str,
) -> tuple:
    molecular = copy.deepcopy(Mol[name_mol])
    print(f"Generate {name_mol}_{distance:.4f}")
    print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}")
    print("original mol", molecular)
    name = f"{name_mol}_{basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

    if "-" in extend_atom:
        if "." in extend_atom:
            extend_atom_1_l = [
                int(i_atom) for i_atom in extend_atom.split("-")[0].split(".")
            ]
            extend_atom_2_l = [
                int(i_atom) for i_atom in extend_atom.split("-")[1].split(".")
            ]
            print(extend_atom_1_l, extend_atom_2_l)
            for extend_i in extend_atom_1_l:
                if extend_i >= len(Mol[name_mol]):
                    print(f"Skip: {name:>40}")
                    return None, name
            for extend_i in extend_atom_2_l:
                if extend_i >= len(Mol[name_mol]):
                    print(f"Skip: {name:>40}")
                    return None, name
            if abs(distance) < 1e-3:
                return None, name
            distance_1_2_array = [
                molecular[extend_atom_2_l[0]][1] - molecular[extend_atom_1_l[0]][1],
                molecular[extend_atom_2_l[0]][2] - molecular[extend_atom_1_l[0]][2],
                molecular[extend_atom_2_l[0]][3] - molecular[extend_atom_1_l[0]][3],
            ]
            distance_1_2 = sum(map(lambda x: x**2, distance_1_2_array)) ** 0.5
            for i in range(1, 4):
                for extend_i in extend_atom_2_l:
                    molecular[extend_i][i] += (
                        distance * distance_1_2_array[i - 1] / distance_1_2
                    )
        else:
            extend_atom_1, extend_atom_2 = map(int, extend_atom.split("-"))
            if extend_atom_1 >= len(Mol[name_mol]) or extend_atom_2 >= len(
                Mol[name_mol]
            ):
                print(f"Skip: {name:>40}")
                return None, name
            if abs(distance) < 1e-3:
                if (extend_atom_1 != 0) and (extend_atom_2 != 1):
                    print(f"Skip: {name:>40}")
                    return None, name
            distance_1_2_array = [
                molecular[extend_atom_2][1] - molecular[extend_atom_1][1],
                molecular[extend_atom_2][2] - molecular[extend_atom_1][2],
                molecular[extend_atom_2][3] - molecular[extend_atom_1][3],
            ]
            distance_1_2 = sum(map(lambda x: x**2, distance_1_2_array)) ** 0.5
            for i in range(1, 4):
                molecular[extend_atom_2][i] += (
                    distance * distance_1_2_array[i - 1] / distance_1_2
                )
    else:
        extend_atom = int(extend_atom)
        if abs(distance) < 1e-3:
            if (extend_atom != 0) or extend_xyz != 1:
                print(f"Skip: {name:>40}")
                return None, name
        if extend_atom >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            return None, name
        molecular[extend_atom][extend_xyz] += distance
    print("extend mol", molecular)
    return molecular, name


def calculate_density_dipole(dm1_scf, df_dict, dft2cc):
    """
    Calculate the density and dipole
    """
    if len(np.shape(dm1_scf)) == 2:
        scf_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0_test,
            dm1_scf,
        )
        cc_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0_test,
            dft2cc.dm1_cc,
        )
        dft_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0_test,
            dft2cc.dm1_dft,
        )
    elif len(np.shape(dm1_scf)) == 3:
        scf_rho_r = np.array(
            [
                pyscf.dft.numint.eval_rho(
                    dft2cc.mol,
                    dft2cc.ao_0_test,
                    dm1_scf[i_spin],
                )
                for i_spin in range(2)
            ]
        )
        dft_rho_r = np.array(
            [
                pyscf.dft.numint.eval_rho(
                    dft2cc.mol,
                    dft2cc.ao_0_test,
                    dft2cc.dm1_dft[i_spin],
                )
                for i_spin in range(2)
            ]
        )
        cc_rho_r = np.array(
            [
                pyscf.dft.numint.eval_rho(
                    dft2cc.mol,
                    dft2cc.ao_0_test,
                    dft2cc.dm1_cc[i_spin],
                )
                for i_spin in range(2)
            ]
        )
    else:
        raise ValueError("dm1_scf shape error")

    error_scf_rho_r = np.sum(np.abs(scf_rho_r - cc_rho_r) * dft2cc.grids_test.weights)
    error_dft_rho_r = np.sum(np.abs(dft_rho_r - cc_rho_r) * dft2cc.grids_test.weights)
    print(
        f"error_scf_rho_r: {error_scf_rho_r:.2e}",
        f"error_dft_rho_r: {error_dft_rho_r:.2e}",
        flush=True,
    )
    df_dict["error_scf_rho_r"].append(error_scf_rho_r)
    df_dict["error_dft_rho_r"].append(error_dft_rho_r)

    for orientation in ["x", "y", "z"]:
        orientation_number = ORIENTATION_NUMBER_DICT[orientation]
        dipole_core = 0
        for i_atom in range(dft2cc.mol.natm):
            dipole_core += (
                dft2cc.mol.atom_charges()[i_atom] * dft2cc.mol.atom_coords()[i_atom][0]
            )

        dipole = dipole_core - np.sum(
            cc_rho_r
            * dft2cc.grids_test.coords[:, orientation_number]
            * dft2cc.grids_test.weights
        )
        dipole_scf = dipole_core - np.sum(
            scf_rho_r
            * dft2cc.grids_test.coords[:, orientation_number]
            * dft2cc.grids_test.weights
        )
        dipole_dft = dipole_core - np.sum(
            dft_rho_r
            * dft2cc.grids_test.coords[:, orientation_number]
            * dft2cc.grids_test.weights
        )

        print(
            f"dipole_{orientation}, scf {dipole_scf - dipole:.4f}, dft {dipole_dft - dipole:.4f}"
        )
        df_dict[f"dipole_{orientation}_diff_scf"].append(dipole_scf - dipole)
        df_dict[f"dipole_{orientation}_diff_dft"].append(dipole_dft - dipole)
    return df_dict


def calculate_force(grad_ai, df_dict, dft2cc):
    """
    Calculate the force
    """
    for orientation in ["x", "y", "z"]:
        orientation_number = ORIENTATION_NUMBER_DICT[orientation]

        error_force_ai = np.linalg.norm(
            grad_ai[:, orientation_number] - dft2cc.grad_ccsd[:, orientation_number],
            ord=1,
        )
        error_force_dft = np.linalg.norm(
            dft2cc.grad_dft[:, orientation_number]
            - dft2cc.grad_ccsd[:, orientation_number],
            ord=1,
        )

        print(
            f"error_force_{orientation}, scf: {error_force_ai:.2e}, "
            f"dft: {error_force_dft:.2e}",
            flush=True,
        )

        df_dict[f"error_force_{orientation}_scf"].append(error_force_ai)
        df_dict[f"error_force_{orientation}_dft"].append(error_force_dft)
    return df_dict
