"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product
from tqdm import tqdm

import torch
import numpy as np
import pyscf
from pyscf.scf.hf import init_guess_by_minao
import opt_einsum as oe

from cadft import add_args, extend, gen_logger
from cadft import CC_DFT_DATA
from cadft.utils import DIIS
from cadft.utils.env_var import DATA_PATH, DATA_SAVE_PATH
from cadft.utils.Grids import Grid
from cadft.utils import ModelDict as ModelDict
from cadft.utils.DataBase import process_input
from cadft.utils.gen_tau import gen_tau_rho


if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init the model
    modeldict = ModelDict(
        load=args.load,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        residual=args.residual,
        device=device,
        precision=args.precision,
        if_mkdir=False,
        load_epoch=args.load_epoch,
    )
    modeldict.load_model()
    modeldict.eval()

    df_dict = {
        "name": [],
        "error_scf_ene": [],
        "error_scf_rho_r": [],
        "dipole_x_diff_scf": [],
        "dipole_y_diff_scf": [],
        "dipole_z_diff_scf": [],
        "time_cc": [],
        "time_dft": [],
    }

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
        molecular, name = extend(
            name_mol, extend_atom, extend_xyz, distance, args.basis
        )
        if molecular is None:
            print(f"Skip: {name:>40}")
            continue

        df_dict["name"].append(name)

        if abs(distance) > 4:
            N_DIIS = 70
        elif abs(distance) > 3:
            N_DIIS = 50
        else:
            N_DIIS = 20

        dft2cc = CC_DFT_DATA(
            molecular,
            name=name,
            basis=args.basis,
            if_basis_str=args.if_basis_str,
        )
        dft2cc.test_mol()
        nocc = dft2cc.mol.nelec[0]
        mdft = pyscf.scf.RKS(dft2cc.mol)

        # 2.1 SCF loop to get the density matrix
        dm1_scf = init_guess_by_minao(dft2cc.mol)

        oe_fock = oe.contract_expression(
            "p,p,pa,pb->ab",
            np.shape(dft2cc.ao_0[:, 0]),
            np.shape(dft2cc.ao_0[:, 0]),
            dft2cc.ao_0,
            dft2cc.ao_0,
            constants=[2, 3],
            optimize="optimal",
        )

        if modeldict.dtype == torch.float32:
            MAX_ERROR_SCF = 1e-5
        else:
            MAX_ERROR_SCF = 1e-8

        diis = DIIS(dft2cc.mol.nao, n=20)
        convergence_count = 0

        for i in range(5000):
            scf_r_3 = pyscf.dft.numint.eval_rho(
                dft2cc.mol, dft2cc.ao_1, dm1_scf, xctype="GGA"
            )
            vxc_scf = modeldict.get_v(scf_r_3, dft2cc.grids)
            exc_b3lyp = pyscf.dft.libxc.eval_xc("lda,vwn", scf_r_3[0])[1][0]
            vxc_scf += exc_b3lyp

            vxc_mat = oe_fock(vxc_scf, dft2cc.grids.weights)
            vj_scf = mdft.get_j(dft2cc.mol, dm1_scf)
            mat_fock = dft2cc.h1e + vj_scf + vxc_mat

            diis.add(
                mat_fock,
                mat_fock @ dm1_scf @ dft2cc.mat_s - dft2cc.mat_s @ dm1_scf @ mat_fock,
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
            if (i > 0) and (error_dm1 < MAX_ERROR_SCF):
                if convergence_count > 5:
                    dm1_scf = dm1_scf_old.copy()
                    break
                else:
                    convergence_count += 1
            else:
                convergence_count = 0

        data = np.load(DATA_PATH / f"data_{name}.npz")
        grids = Grid(dft2cc.mol)
        ao_value = pyscf.dft.numint.eval_ao(dft2cc.mol, grids.coords, deriv=2)
        inv_r_3 = pyscf.dft.numint.eval_rho(
            dft2cc.mol, ao_value[:4, :, :], dm1_scf, xctype="GGA"
        )
        inv_r = inv_r_3[0] + 1e-12
        evxc_lda = pyscf.dft.libxc.eval_xc("lda,vwn", inv_r)
        data_grids_norm = process_input(inv_r_3, grids)

        ao_0 = ao_value[0, :, :]
        ao_2_diag = ao_value[4, :, :] + ao_value[7, :, :] + ao_value[9, :, :]
        oe_tau_rho = oe.contract_expression(
            "pm,m,n,pn->p",
            ao_0,
            (mo_scf.shape[1],),
            (mo_scf.shape[1],),
            ao_2_diag,
            constants=[0, 3],
            optimize="optimal",
        )

        n_slice_grids = torch.cuda.mem_get_info()[0] // 4 // 8 // dft2cc.mol.nao**2
        n_batchs_grids = len(grids.coords) // n_slice_grids + 1

        exc_grids_fake1 = np.zeros_like(inv_r)
        rho_cc = grids.matrix_to_vector(data["rho_cc"])
        int1e_grids = dft2cc.mol.intor("int1e_grids", grids=grids.coords)

        for i_batch_grids in range(n_batchs_grids):
            ngrids_slice_i = (
                n_slice_grids
                if i_batch_grids != n_slice_grids - 1
                else len(grids.coords) - n_slice_grids * i_batch_grids
            )
            i_slice_grids = slice(
                n_slice_grids * i_batch_grids,
                n_slice_grids * i_batch_grids + ngrids_slice_i,
            )
            vele = np.einsum(
                "pij,ij->p",
                int1e_grids[i_slice_grids, :, :],
                data["dm_cc"],
            )
            exc_grids_fake1[i_slice_grids] += vele * (
                rho_cc[i_slice_grids] - inv_r[i_slice_grids]
            )

        for i, coord in enumerate(tqdm(grids.coords)):
            for i_atom in range(dft2cc.mol.natm):
                distance = np.linalg.norm(dft2cc.mol.atom_coords()[i_atom] - coord)
                if distance > 1e-2:
                    exc_grids_fake1[i] -= (
                        (rho_cc[i] - inv_r[i])
                        * dft2cc.mol.atom_charges()[i_atom]
                        / distance
                    )
                else:
                    exc_grids_fake1[i] -= (
                        (rho_cc[i] - inv_r[i])
                        * dft2cc.mol.atom_charges()[i_atom]
                        / 1e-2
                    )

        tau_rho_ks = gen_tau_rho(
            inv_r,
            mo_scf[:, :nocc],
            2 * np.ones(nocc),
            oe_tau_rho,
            backend="torch",
        )
        tau_rho_wf = np.load(DATA_SAVE_PATH / f"{name}" / "tau_rho_wf.npy")

        np.savez_compressed(
            DATA_PATH / f"data_{name}.npz",
            dm_cc=data["dm_cc"],
            dm_inv=data["dm_inv"],
            rho_cc=data["rho_cc"],
            weights=data["weights"],
            vxc=data["vxc"],
            vxc_b3lyp=data["vxc_b3lyp"],
            vxc1_b3lyp=data["vxc1_b3lyp"],
            exc=data["exc"],
            exc_real=data["exc_real"],
            exc_tr_b3lyp=data["exc_tr_b3lyp"],
            exc1_tr_b3lyp=data["exc1_tr_b3lyp"],
            exc_tr=data["exc_tr"],
            exc1_tr=data["exc1_tr"],
            coords_x=data["coords_x"],
            coords_y=data["coords_y"],
            coords_z=data["coords_z"],
            rho_inv=grids.vector_to_matrix(inv_r),
            rho_inv_4_norm=data_grids_norm,
            exc1_tr_lda=data["exc_real"]
            + grids.vector_to_matrix(
                0.1 * (exc_grids_fake1 + tau_rho_wf - tau_rho_ks) / inv_r - evxc_lda[0]
            ),
            vxc1_lda=data["vxc"] - grids.vector_to_matrix(evxc_lda[1][0]),
        )

    # if "openshell" in name_mol:
    #     test_uks_data(args, molecular, name, modeldict)
    # else:
    #     dm1_scf = test_rks_data(
    #         args, molecular, name, modeldict, n_diis=N_DIIS, dm1_scf=dm1_scf
    #     )
