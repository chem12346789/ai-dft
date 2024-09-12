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
from cadft.utils import calculate_density_dipole

AU2KCALMOL = 627.5096080306


def test_uks(
    args,
    molecular,
    name,
    modeldict,
    df_dict: dict,
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
        spin=0,
    )
    dft2cc.utest_mol(level=args.level)
    nocc = dft2cc.mol.nelec
    mdft = pyscf.scf.UKS(dft2cc.mol)

    # 1.1 SCF loop to get the density matrix
    time_start = timer()

    # dm1_scf = dft2cc.dm1_dft
    if dm_guess is None:
        dm1_scf = init_guess_by_minao(dft2cc.mol)
        dm1_scf = np.array([dm1_scf / 2, dm1_scf / 2])
    else:
        dm1_scf = dm_guess.copy()

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
        max_error_scf = 1e-4
    else:
        max_error_scf = 1e-8

    diis = (DIIS(dft2cc.mol.nao, n=12), DIIS(dft2cc.mol.nao, n=12))

    for i in range(100):
        if modeldict.input_size == 1:
            scf_rho_r = np.array(
                [
                    pyscf.dft.numint.eval_rho(
                        dft2cc.mol,
                        dft2cc.ao_0,
                        2 * dm1_scf[i_spin],
                    )
                    for i_spin in range(2)
                ]
            )
            vxc_scf = np.array(
                [
                    modeldict.get_v(scf_rho_r[i_spin], dft2cc.grids)
                    + pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r[i_spin])[1][0]
                    for i_spin in range(2)
                ]
            )
        elif modeldict.input_size == 4:
            scf_rho_r3 = np.array(
                [
                    pyscf.dft.numint.eval_rho(
                        dft2cc.mol,
                        dft2cc.ao_1,
                        2 * dm1_scf[i_spin],
                        xctype="GGA",
                    )
                    for i_spin in range(2)
                ]
            )
            vxc_scf = np.array(
                [
                    modeldict.get_v(scf_rho_r3[i_spin], dft2cc.grids)
                    + pyscf.dft.libxc.eval_xc("b3lyp", scf_rho_r3[i_spin])[0]
                    for i_spin in range(2)
                ]
            )

        vxc_mat = np.array(
            [oe_fock(vxc_scf[i_spin], dft2cc.grids.weights) for i_spin in range(2)]
        )
        vj_scf = mdft.get_j(dft2cc.mol, dm1_scf[0] + dm1_scf[1])
        mat_fock = np.array(
            [dft2cc.h1e + vj_scf + vxc_mat[i_spin] for i_spin in range(2)]
        )

        dm1_scf_old = dm1_scf.copy()
        for i_spin in range(2):
            diis[i_spin].add(
                mat_fock[i_spin],
                dft2cc.mat_s @ dm1_scf[i_spin] @ mat_fock[i_spin]
                - mat_fock[i_spin] @ dm1_scf[i_spin] @ dft2cc.mat_s,
            )
            mat_fock[i_spin] = diis[i_spin].hybrid()
            _, mo_scf = np.linalg.eigh(dft2cc.mat_hs @ mat_fock[i_spin] @ dft2cc.mat_hs)
            mo_scf = dft2cc.mat_hs @ mo_scf

            dm1_scf[i_spin] = mo_scf[:, : nocc[i_spin]] @ mo_scf[:, : nocc[i_spin]].T
        error_dm1 = np.linalg.norm(dm1_scf - dm1_scf_old)

        print(
            f"step:{i:<8}",
            f"dm: {error_dm1::<10.5e}",
        )
        if (i > 0) and (error_dm1 < max_error_scf):
            dm1_scf = dm1_scf_old.copy()
            break

    del oe_fock
    gc.collect()
    torch.cuda.empty_cache()

    # 2 check
    # 2.1 check the difference of density (on grids) and dipole
    if hasattr(dft2cc, "time_cc"):
        print(
            f"cc: {dft2cc.time_cc:.2f}s, aidft: {(timer() - time_start):.2f}s",
            flush=True,
        )
        print(dft2cc.time_cc)
        print(dft2cc.time_dft)
        df_dict["time_cc"].append(dft2cc.time_cc)
        df_dict["time_dft"].append(dft2cc.time_dft)
        df_dict["time_ai"].append(timer() - time_start)
    else:
        df_dict["time_cc"].append(-1)
        df_dict["time_dft"].append(-1)
        df_dict["time_ai"].append(timer() - time_start)

    # 2.3 check the difference of energy (total)
    df_dict = calculate_density_dipole(dm1_scf, df_dict, dft2cc)

    if modeldict.input_size == 1:
        exc_scf = np.array(
            [
                modeldict.get_e(scf_rho_r[i_spin], dft2cc.grids)
                + pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r[i_spin])[0]
                for i_spin in range(2)
            ]
        )
    elif modeldict.input_size == 4:
        scf_rho_r3 = np.array(
            [
                pyscf.dft.numint.eval_rho(
                    dft2cc.mol,
                    dft2cc.ao_1,
                    2 * dm1_scf[i_spin],
                    xctype="GGA",
                )
                for i_spin in range(2)
            ]
        )
        exc_scf = np.array(
            [
                modeldict.get_e(scf_rho_r3[i_spin], dft2cc.grids)
                + pyscf.dft.libxc.eval_xc("b3lyp", scf_rho_r3[i_spin])[0]
                for i_spin in range(2)
            ]
        )

    scf_rho_r = np.array(
        [
            pyscf.dft.numint.eval_rho(dft2cc.mol, dft2cc.ao_0, dm1_scf[i_spin])
            for i_spin in range(2)
        ]
    )
    ene_scf = (
        oe.contract("ij,ji->", dft2cc.h1e, dm1_scf[0] + dm1_scf[1])
        + 0.5 * oe.contract("ij,ji->", vj_scf, dm1_scf[0] + dm1_scf[1])
        + dft2cc.mol.energy_nuc()
        + np.sum(exc_scf * scf_rho_r * dft2cc.grids.weights)
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

    # check the difference of force
    for orientation in ["x", "y", "z"]:
        df_dict[f"error_force_{orientation}_scf"].append(-1)
        df_dict[f"error_force_{orientation}_dft"].append(-1)

    print(df_dict)
    df = pd.DataFrame(df_dict)
    df.to_csv(
        Path(
            f"{MAIN_PATH}/validate/ccdft_{args.load}_{args.hidden_size}_{args.num_layers}_{args.residual}"
        ),
        index=False,
    )
    return dm1_scf
