import gc
from pathlib import Path
from timeit import default_timer as timer

import pyscf
from pyscf.scf.hf import init_guess_by_minao
import pandas as pd
import torch
import numpy as np
import opt_einsum as oe

from cadft import CC_DFT_DATA
from cadft.utils import DIIS
from cadft.utils import MAIN_PATH

AU2KCALMOL = 627.5096080306


def test_rks(
    args,
    molecular,
    name,
    modeldict,
    df_dict: dict,
    n_diis: int = 10,
    dm_guess=None,
):
    """
    Test the model. Restrict Khon-Sham (no spin).
    """
    # 2.0 Prepare
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
    time_start = timer()

    if dm_guess is None:
        dm1_scf = init_guess_by_minao(dft2cc.mol)
    else:
        dm1_scf = dm_guess.copy()
    # dm1_scf = dft2cc.dm1_dft

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
        max_error_scf = 1e-5
    else:
        max_error_scf = 1e-8

    diis = DIIS(dft2cc.mol.nao, n=n_diis)
    converge_setp = 0
    max_steps_converge = 5

    for i in range(500):
        scf_r_3 = pyscf.dft.numint.eval_rho(
            dft2cc.mol, dft2cc.ao_1, dm1_scf, xctype="GGA"
        )
        vxc_scf = modeldict.get_v(scf_r_3, dft2cc.grids)
        exc_b3lyp = pyscf.dft.libxc.eval_xc("lda,vwn", scf_r_3[0])[1][0]
        # exc_b3lyp = pyscf.dft.libxc.eval_xc("b3lyp", scf_r_3)[0]
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
        if (i > 0) and (error_dm1 < max_error_scf):
            if converge_setp > max_steps_converge:
                dm1_scf = dm1_scf_old.copy()
                break
            else:
                converge_setp += 1
        else:
            converge_setp = 0

    # 2.2 Check the difference of density (on grids) and dipole
    print(
        f"cc: {dft2cc.time_cc:.2f}s, aidft: {(timer() - time_start):.2f}s",
        flush=True,
    )
    df_dict["time_cc"].append(dft2cc.time_cc)
    df_dict["time_dft"].append(timer() - time_start)

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
        f"error_scf_rho_r: {error_scf_rho_r:.2e}",
        f"error_dft_rho_r: {error_dft_rho_r:.2e}",
        flush=True,
    )
    df_dict["error_scf_rho_r"].append(error_scf_rho_r)
    df_dict["error_dft_rho_r"].append(error_dft_rho_r)

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
    df_dict["dipole_x_diff_scf"].append(dipole_x_scf - dipole_x)
    df_dict["dipole_y_diff_scf"].append(dipole_y_scf - dipole_y)
    df_dict["dipole_z_diff_scf"].append(dipole_z_scf - dipole_z)
    df_dict["dipole_x_diff_dft"].append(dipole_x_dft - dipole_x)
    df_dict["dipole_y_diff_dft"].append(dipole_y_dft - dipole_y)
    df_dict["dipole_z_diff_dft"].append(dipole_z_dft - dipole_z)

    # 2.3 check the difference of energy (total)

    scf_r_3 = pyscf.dft.numint.eval_rho(dft2cc.mol, dft2cc.ao_1, dm1_scf, xctype="GGA")
    exc_scf = modeldict.get_e(scf_r_3, dft2cc.grids)
    exc_ene = np.sum(exc_scf * scf_r_3[0] * dft2cc.grids.weights)
    exc_b3lyp = pyscf.dft.libxc.eval_xc("lda,vwn", scf_r_3[0])[0]
    # exc_b3lyp = pyscf.dft.libxc.eval_xc("b3lyp", scf_r_3)[0]
    exc_ene += np.sum(exc_b3lyp * scf_r_3[0] * dft2cc.grids.weights)

    ene_scf = (
        oe.contract("ij,ji->", dft2cc.h1e, dm1_scf)
        + 0.5 * oe.contract("ij,ji->", vj_scf, dm1_scf)
        + dft2cc.mol.energy_nuc()
        + exc_ene
    )
    error_ene_scf = AU2KCALMOL * (ene_scf - dft2cc.e_cc)
    error_ene_dft = AU2KCALMOL * (dft2cc.e_dft - dft2cc.e_cc)
    print(
        f"error_scf_ene: {error_ene_scf:.2e}, error_dft_ene: {error_ene_dft:.2e}",
        flush=True,
    )

    df_dict["error_scf_ene"].append(error_ene_scf)
    df_dict["error_dft_ene"].append(error_ene_dft)
    df_dict["abs_scf_ene"].append(AU2KCALMOL * ene_scf)
    df_dict["abs_dft_ene"].append(AU2KCALMOL * dft2cc.e_dft)
    df_dict["abs_cc_ene"].append(AU2KCALMOL * dft2cc.e_cc)

    df = pd.DataFrame(df_dict)
    df.to_csv(
        Path(
            f"{MAIN_PATH}/validate/ccdft_{args.load}_{args.hidden_size}_{args.num_layers}_{args.residual}"
        ),
        index=False,
    )
    return dm1_scf
