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
from cadft.utils import MAIN_PATH, DATA_PATH
from cadft.utils import calculate_density_dipole

AU2KCALMOL = 627.5096080306


def test_rks(
    args,
    molecular,
    name,
    modeldict,
    df_dict: dict,
    n_diis: int = 10,
    dm_guess=None,
    from_data=False,
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

    if from_data:
        if (DATA_PATH / f"data_{name}.npz").exists():
            data_real = np.load(DATA_PATH / f"data_{name}.npz")
        else:
            print(f"No file: {name:>40}")
            return

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
        scf_rho_r = pyscf.dft.numint.eval_rho(dft2cc.mol, dft2cc.ao_0, dm1_scf)
        if from_data:
            middle_mat = data_real["vxc"]
            vxc_scf = dft2cc.grids.matrix_to_vector(middle_mat)
        else:
            vxc_scf = modeldict.get_v(scf_rho_r, dft2cc.grids)
            vxc_b3lyp = pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r)[1][0]
            vxc_scf += vxc_b3lyp

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

    del oe_fock
    gc.collect()
    torch.cuda.empty_cache()

    if hasattr(dft2cc, "time_cc"):
        print(
            f"cc: {dft2cc.time_cc:.2f}s, aidft: {(timer() - time_start):.2f}s",
            flush=True,
        )
        df_dict["time_cc"].append(dft2cc.time_cc)
        df_dict["time_dft"].append(dft2cc.time_dft)
        df_dict["time_ai"].append(timer() - time_start)
    else:
        df_dict["time_cc"].append(-1)
        df_dict["time_dft"].append(-1)
        df_dict["time_ai"].append(timer() - time_start)

    # 2.2 Check the difference of density (on grids) and dipole
    df_dict = calculate_density_dipole(dm1_scf, df_dict, dft2cc)

    # 2.3 check the difference of energy (total)
    scf_rho_r = pyscf.dft.numint.eval_rho(
        dft2cc.mol,
        dft2cc.ao_0,
        dm1_scf,
    )
    if from_data:
        output_mat = data_real["exc1_tr_lda"]
        exc_scf = (
            dft2cc.grids.matrix_to_vector(output_mat)
            + pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r)[0]
        )
    else:
        exc_scf = (
            modeldict.get_e(scf_rho_r, dft2cc.grids)
            + pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r)[0]
        )

    ene_scf = (
        oe.contract("ij,ji->", dft2cc.h1e, dm1_scf)
        + 0.5 * oe.contract("ij,ji->", vj_scf, dm1_scf)
        + np.sum(exc_scf * scf_rho_r * dft2cc.grids.weights)
        + dft2cc.mol.energy_nuc()
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
