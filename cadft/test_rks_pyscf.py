"""
    Test the model. Restrict Khon-Sham (spin). Modify the pyscf code.
"""

import gc
from pathlib import Path
from timeit import default_timer as timer
import types

import pyscf
from pyscf import lib
from pyscf.dft import numint

import pandas as pd
import torch
import numpy as np
import opt_einsum as oe

from cadft import CC_DFT_DATA, cc_dft_data
from cadft.utils import MAIN_PATH, DATA_PATH
from cadft.utils import calculate_density_dipole, calculate_force

AU2KCALMOL = 627.5096080306


def test_rks_pyscf(
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

    from_data = getattr(args, "from_data", False)
    require_grad = getattr(args, "require_grad", False)

    # 2.0 Prepare
    dft2cc = CC_DFT_DATA(
        molecular,
        name=name,
        basis=args.basis,
        if_basis_str=args.if_basis_str,
    )
    dft2cc.test_mol(require_grad)
    mdft = pyscf.scf.RKS(dft2cc.mol)

    if from_data:
        if (DATA_PATH / f"data_{name}.npz").exists():
            data_real = np.load(DATA_PATH / f"data_{name}.npz")
        else:
            print(f"No file: {name:>40}")
            return

    # 2.1 SCF loop to get the density matrix
    time_start = timer()

    oe_fock = oe.contract_expression(
        "p,p,pa,pb->ab",
        np.shape(dft2cc.ao_0[:, 0]),
        np.shape(dft2cc.ao_0[:, 0]),
        dft2cc.ao_0,
        dft2cc.ao_0,
        constants=[2, 3],
        optimize="optimal",
    )

    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks.mol
        if dm is None:
            dm = ks.make_rdm1()

        scf_rho_r = pyscf.dft.numint.eval_rho(dft2cc.mol, dft2cc.ao_0, dm)
        vexc_lda = pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r)
        # vxc_mat = oe_fock(vexc_lda[1][0], dft2cc.grids.weights)
        # exc_ene = np.sum(vexc_lda[0] * scf_rho_r * dft2cc.grids.weights)

        if from_data:
            vxc_scf = modeldict.get_v(scf_rho_r, dft2cc.grids) + vexc_lda[1][0]
        else:
            middle_mat = data_real["vxc"]
            vxc_scf = dft2cc.grids.matrix_to_vector(middle_mat)

        if from_data:
            exc_scf = modeldict.get_e(scf_rho_r, dft2cc.grids) + vexc_lda[0]
        else:
            output_mat = data_real["exc1_tr_lda"]
            exc_scf = dft2cc.grids.matrix_to_vector(output_mat) + vexc_lda[0]
        vxc_mat = oe_fock(vxc_scf, dft2cc.grids.weights)
        exc_ene = np.sum(exc_scf * scf_rho_r * dft2cc.grids.weights)

        vk = None
        if (
            ks._eri is None
            and ks.direct_scf
            and getattr(vhf_last, "vj", None) is not None
        ):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm, hermi)
        vxc_mat += vj

        ecoul = np.einsum("ij,ji", dm, vj).real * 0.5
        vxc_mat = lib.tag_array(vxc_mat, ecoul=ecoul, exc=exc_ene, vj=vj, vk=vk)
        return vxc_mat

    mdft.get_veff = types.MethodType(get_veff, mdft)
    mdft.xc = "lda,vwn"
    if args.precision == "float32":
        mdft.conv_tol = 1e-5
    elif args.precision == "float64":
        mdft.conv_tol = 1e-8
    mdft.diis_space = n_diis
    mdft.max_cycle = 250
    mdft.level_shift = 0
    if dm_guess is not None:
        mdft.run(dm0=dm_guess)
    else:
        mdft.run()
    dm1_scf = mdft.make_rdm1()
    print("Done SCF", flush=True)

    def get_vxc(dms):
        """
        Modification of the get_vxc function in pyscf.grad.rks.py
        https://pyscf.org/_modules/pyscf/grad/rks.html
        """
        # print("Using modified get_vxc", flush=True)
        scf_rho_r = pyscf.dft.numint.eval_rho(dft2cc.mol, dft2cc.ao_0, dms)
        vexc_lda = pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r)

        if from_data:
            vxc = modeldict.get_v(scf_rho_r, dft2cc.grids) + vexc_lda[1][0]
        else:
            middle_mat = data_real["vxc"]
            vxc = dft2cc.grids.matrix_to_vector(middle_mat)

        wv = dft2cc.grids.weights * vxc
        aow = np.einsum("gi,g->gi", dft2cc.ao_1[0], wv)
        vmat = np.array([dft2cc.ao_1[i].T @ aow for i in range(1, 4)])
        exc = None
        # - sign because nabla_X = -nabla_x
        return exc, -vmat

    if require_grad:

        def get_veff_grad(ks_grad, mol=None, dm=None):
            """
            Modification of the get_veff function in pyscf.grad.rks.py
            https://pyscf.org/_modules/pyscf/grad/rks.html
            """
            # print("Using modified get_veff_grad", flush=True)
            if mol is None:
                mol = ks_grad.mol
            if dm is None:
                dm = ks_grad.base.make_rdm1()

            exc, vxc = get_vxc(dm)
            vj = ks_grad.get_j(mol, dm)
            vxc += vj

            return lib.tag_array(vxc, exc1_grid=exc)

        g = mdft.nuc_grad_method()
        g.get_veff = types.MethodType(get_veff_grad, g)
        grad_ai = g.kernel()
        print("If net force", np.sum(grad_ai, axis=0))

    del oe_fock
    gc.collect()
    torch.cuda.empty_cache()

    # 2.2 Check the difference of density (on grids) and dipole
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

    df_dict = calculate_density_dipole(dm1_scf, df_dict, dft2cc)

    # 2.3 check the difference of energy (total)
    ene_scf = mdft.e_tot
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
    if require_grad:
        df_dict = calculate_force(grad_ai, df_dict, dft2cc)
    else:
        for orientation in ["x", "y", "z"]:
            df_dict[f"error_force_{orientation}_scf"].append(-1)
            df_dict[f"error_force_{orientation}_dft"].append(-1)

    df = pd.DataFrame(df_dict)
    df.to_csv(
        Path(
            f"{MAIN_PATH}/validate/ccdft_{args.load}_{args.hidden_size}_{args.num_layers}_{args.residual}"
        ),
        index=False,
    )
    return dm1_scf
