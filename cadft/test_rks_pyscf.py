"""
    Test the model. Restrict Khon-Sham (spin). Modify the pyscf code.
"""

import gc
from timeit import default_timer as timer
import types

import pyscf
from pyscf import lib

import pandas as pd
import torch
import numpy as np

from cadft import CC_DFT_DATA
from cadft.utils import MAIN_PATH, DATA_PATH
from cadft.utils import calculate_density_dipole, calculate_force
from cadft.utils import rotate

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
    generate_data = getattr(args, "generate_data", False)
    require_grad = getattr(args, "require_grad", False)

    if from_data:
        if (DATA_PATH / f"data_{name}.npz").exists():
            data_real = np.load(DATA_PATH / f"data_{name}.npz")
        else:
            print(f"No file: {name:>40}")
            return

    df_dict["name"].append(name)

    # 2.0 Prepare
    rotate(molecular, rotation="r", verbose=True)
    dft2cc = CC_DFT_DATA(
        molecular,
        name=name,
        basis=args.basis,
        if_basis_str=args.if_basis_str,
    )
    dft2cc.test_mol(require_grad)
    mdft = pyscf.scf.RKS(dft2cc.mol)

    # 2.1 SCF loop to get the density matrix
    time_start = timer()

    def get_veff_modified(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks.mol

        if dm is None:
            dm = ks.make_rdm1()

        if mol.spin == 0:
            vhf = pyscf.dft.rks.get_veff(ks, mol, dm, dm_last, vhf_last, hermi)
        else:
            vhf = pyscf.dft.uks.get_veff(ks, mol, dm, dm_last, vhf_last, hermi)

        ecoul = vhf.ecoul
        exc = vhf.exc
        vj = vhf.vj
        vk = vhf.vk

        if from_data:
            scf_rho_r = pyscf.dft.numint.eval_rho(dft2cc.mol, dft2cc.ao_0, dm)
            middle_mat = data_real["vxc1_lda"]
            vxc_scf = dft2cc.grids.matrix_to_vector(middle_mat)
            output_mat = data_real["exc1_tr_lda"]
            exc_scf = dft2cc.grids.matrix_to_vector(output_mat)
            vhf += pyscf.dft.numint.eval_mat(
                dft2cc.mol, dft2cc.ao_0, dft2cc.grids.weights, vxc_scf, vxc_scf
            )
            exc += np.sum(exc_scf * scf_rho_r * dft2cc.grids.weights)
        else:
            vxc_scf = modeldict.get_v(ks, dft2cc.grids, dm)
            vhf += pyscf.dft.numint.eval_mat(
                dft2cc.mol, dft2cc.ao_0, dft2cc.grids.weights, vxc_scf, vxc_scf
            )
            exc += modeldict.get_e(ks, dft2cc.grids, dm)

        # rho_diff = ni.eval_rho(dft2cc.mol, dft2cc.ao_0, dm - dft2cc.dm1_cc)
        # v_p = pyscf.dft.numint.eval_mat(
        #     dft2cc.mol, dft2cc.ao_0, dft2cc.grids.weights, rho_diff, rho_diff
        # )
        # vxc += 100 * v_p

        vxc = lib.tag_array(vhf, ecoul=ecoul, exc=exc, vj=vj, vk=vk)

        return vxc

    mdft.get_veff = types.MethodType(get_veff_modified, mdft)
    mdft.xc = "lda,vwn"
    if args.precision == "float32":
        mdft.conv_tol = 1e-4
    elif args.precision == "float64":
        mdft.conv_tol = 1e-6
    mdft.diis_space = n_diis
    mdft.DIIS = pyscf.scf.CDIIS
    mdft.max_cycle = 250
    mdft.level_shift = 0
    if dm_guess is not None:
        mdft.init_guess = dm_guess
        mdft.run()
    else:
        mdft.run()
    dm1_scf = mdft.make_rdm1()
    print("Done SCF", flush=True)

    if require_grad:

        def get_vxc(ks_grad, dms):
            """
            Modification of the get_vxc function in pyscf.grad.rks.py
            https://pyscf.org/_modules/pyscf/grad/rks.html
            """
            # print("Using modified get_vxc", flush=True)
            mf = ks_grad.base
            ni = mf._numint

            scf_rho_r = ni.eval_rho(dft2cc.mol, dft2cc.ao_0, dms)
            vexc_lda = pyscf.dft.libxc.eval_xc("lda,vwn", scf_rho_r)

            if from_data:
                middle_mat = data_real["vxc"]
                vxc_scf = dft2cc.grids.matrix_to_vector(middle_mat)
            else:
                vxc_scf = modeldict.get_v(ks_grad, dft2cc.grids, dms)
                vxc_scf += vexc_lda[1][0]

            wv = dft2cc.grids.weights * vxc_scf
            aow = np.einsum("gi,g->gi", dft2cc.ao_1[0], wv)
            vmat = np.array([dft2cc.ao_1[i].T @ aow for i in range(1, 4)])
            exc = None
            # - sign because nabla_X = -nabla_x
            return exc, -vmat

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

            exc, vxc = get_vxc(ks_grad, dm)
            vj = ks_grad.get_j(mol, dm)
            vxc += vj

            return lib.tag_array(vxc, exc1_grid=exc)

        g = mdft.nuc_grad_method()
        g.get_veff = types.MethodType(get_veff_grad, g)
        grad_ai = g.kernel()
        print("If net force", np.sum(grad_ai, axis=0))

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
    if generate_data:
        csv_path = (
            MAIN_PATH
            / f"validate/ccdft_{args.load}_{args.hidden_size}_{args.num_layers}_{args.residual}_generate_data"
        )
    else:
        csv_path = (
            MAIN_PATH
            / f"validate/ccdft_{args.load}_{args.hidden_size}_{args.num_layers}_{args.residual}"
        )
    print(csv_path)
    df.to_csv(csv_path, index=False)

    if generate_data:
        raise NotImplementedError("Generate data is not implemented yet.")
    return dm1_scf
