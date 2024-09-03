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

from cadft import CC_DFT_DATA, extend, add_args, gen_logger
from cadft.utils import MAIN_PATH, DATA_PATH
from cadft.utils import DIIS

AU2KCALMOL = 627.5096080306

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
        index_dict = {}
        for i_atom in ["H", "C"]:
            index_dict[i_atom] = []

        print(f"Generate {name_mol}_{distance:.4f}")
        molecular, name = extend(
            name_mol, extend_atom, extend_xyz, distance, args.basis
        )
        for i in range(len(molecular)):
            index_dict[molecular[i][0]].append(i)

        if molecular is None:
            print(f"Skip: {name:>40}")
            continue

        if (DATA_PATH / f"data_{name}.npz").exists():
            data_real = np.load(DATA_PATH / f"data_{name}.npz")
        else:
            print(f"No file: {name:>40}")
            continue
        name_list.append(name)

        dft2cc = CC_DFT_DATA(
            molecular,
            name=name,
            basis=args.basis,
            if_basis_str=args.if_basis_str,
        )
        dft2cc.test_mol()
        # dft2cc.test_mol(data_real["dm_cc"], data_real["e_cc"])
        nocc = dft2cc.mol.nelec[0]
        mdft = pyscf.scf.RKS(dft2cc.mol)

        # 1.1 SCF loop to get the density matrix
        time_start = timer()

        # dm1_scf = dft2cc.dm1_dft
        dm1_scf = data_real["dm_inv"]
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
            max_error_scf = 1e-5
        else:
            max_error_scf = 1e-8

        diis = DIIS(dft2cc.mol.nao, n=8)

        for i in range(100):
            middle_mat = data_real["vxc"]
            vxc_scf = dft2cc.grids.matrix_to_vector(middle_mat)

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
        error_scf_rho_r = np.sum(
            np.abs(scf_rho_r - cc_rho_r) * dft2cc.grids_test.weights
        )
        error_dft_rho_r = np.sum(
            np.abs(dft_rho_r - cc_rho_r) * dft2cc.grids_test.weights
        )
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
            cc_rho_r * dft2cc.grids_test.coords[:, 0] * dft2cc.grids_test.weights
        )
        dipole_x_scf = dipole_x_core - np.sum(
            scf_rho_r * dft2cc.grids_test.coords[:, 0] * dft2cc.grids_test.weights
        )
        dipole_x_dft = dipole_x_core - np.sum(
            dft_rho_r * dft2cc.grids_test.coords[:, 0] * dft2cc.grids_test.weights
        )

        dipole_y_core = 0
        for i_atom in range(dft2cc.mol.natm):
            dipole_y_core += (
                dft2cc.mol.atom_charges()[i_atom] * dft2cc.mol.atom_coords()[i_atom][1]
            )
        dipole_y = dipole_y_core - np.sum(
            cc_rho_r * dft2cc.grids_test.coords[:, 1] * dft2cc.grids_test.weights
        )
        dipole_y_scf = dipole_y_core - np.sum(
            scf_rho_r * dft2cc.grids_test.coords[:, 1] * dft2cc.grids_test.weights
        )
        dipole_y_dft = dipole_y_core - np.sum(
            dft_rho_r * dft2cc.grids_test.coords[:, 1] * dft2cc.grids_test.weights
        )

        dipole_z_core = 0
        for i_atom in range(dft2cc.mol.natm):
            dipole_z_core += (
                dft2cc.mol.atom_charges()[i_atom] * dft2cc.mol.atom_coords()[i_atom][2]
            )
        dipole_z = dipole_z_core - np.sum(
            cc_rho_r * dft2cc.grids_test.coords[:, 2] * dft2cc.grids_test.weights
        )
        dipole_z_scf = dipole_z_core - np.sum(
            scf_rho_r * dft2cc.grids_test.coords[:, 2] * dft2cc.grids_test.weights
        )
        dipole_z_dft = dipole_z_core - np.sum(
            dft_rho_r * dft2cc.grids_test.coords[:, 2] * dft2cc.grids_test.weights
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

        inv_r = pyscf.dft.numint.eval_rho(dft2cc.mol, dft2cc.ao_0, dm1_scf)
        output_mat = data_real["exc1_tr_lda"]
        output_mat_exc = output_mat * dft2cc.grids.vector_to_matrix(
            inv_r * dft2cc.grids.weights
        )
        exc_b3lyp = pyscf.dft.libxc.eval_xc("lda,vwn", inv_r)[0]
        b3lyp_ene = np.sum(exc_b3lyp * inv_r * dft2cc.grids.weights)

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
