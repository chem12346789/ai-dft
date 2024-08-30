import json
import gc
from pathlib import Path
from itertools import product
from re import A

import numpy as np
from tqdm import tqdm
import torch
import pyscf
import scipy.linalg as LA
import opt_einsum as oe

from cadft.utils.gen_tau import gen_taup_rho, gen_tau_rho
from cadft.utils.Grids import Grid
from cadft.utils.env_var import DATA_PATH, DATA_SAVE_PATH
from cadft.utils.diis import DIIS
from cadft.utils.DataBase import process_input

AU2KJMOL = 2625.5


def umrks_diis(self, frac_old, load_inv=True, diis_n=15,):
    """
    Generate 1-RDM.
    """
    self.data_save_path = DATA_SAVE_PATH / f"{self.name}"
    Path(self.data_save_path).mkdir(parents=True, exist_ok=True)
    n_slices = 150

    # self.mol.spin = 4
    self.mol.verbose = 4

    mdft = pyscf.scf.UKS(self.mol)
    mdft.xc = "b3lyp"
    mdft.kernel()
    mdft.stability(external=True)
    dm1_dft = mdft.make_rdm1(ao_repr=True)

    mf = pyscf.scf.UHF(self.mol)
    mf.kernel()
    mf.stability(external=True)
    # mycc = pyscf.ci.UCISD(mf)
    mycc = pyscf.cc.UCCSD(mf)
    mycc.max_cycle = 100
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-8
    # mycc.direct = True
    # mycc.incore_complete = True
    # mycc.async_io = False
    mycc.kernel()

    h1e = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
    mo = mf.mo_coeff
    mo_cc = mycc.mo_coeff
    nocc = self.mol.nelec
    nao = self.mol.nao
    print(f"nocc: {nocc}")

    n_batchs = self.mol.nao // n_slices + 1

    mat_s = self.mol.intor("int1e_ovlp")
    mat_hs = LA.fractional_matrix_power(mat_s, -0.5).real

    dm1_cc = np.array(mycc.make_rdm1(ao_repr=True))
    e_cc = mycc.e_tot

    grids = Grid(self.mol)
    coords = grids.coords
    weights = grids.weights
    ao_value = pyscf.dft.numint.eval_ao(self.mol, coords, deriv=2)

    ao_0 = ao_value[0, :, :]
    ao_1 = ao_value[1:4, :, :]
    ao_2_diag = ao_value[4, :, :] + ao_value[7, :, :] + ao_value[9, :, :]

    # 25% total memory for int1egrids
    n_slice_grids = torch.cuda.mem_get_info()[0] // 4 // 8 // self.mol.nao**2
    n_batchs_grids = len(coords) // n_slice_grids + 1
    print(
        f"n_batchs_grids: {n_batchs_grids}. n_slice_grids: {n_slice_grids}, will consume about {n_slice_grids * self.mol.nao**2 * 8 / 1024**3:.2f} GB memory."
    )

    oe_taup_rho = oe.contract_expression(
        "pm,m,n,kpn->pk",
        ao_0,
        (nao,),
        (nao,),
        ao_1,
        constants=[0, 3],
        optimize="optimal",
    )

    oe_tau_rho = oe.contract_expression(
        "pm,m,n,pn->p",
        ao_0,
        (nao,),
        (nao,),
        ao_2_diag,
        constants=[0, 3],
        optimize="optimal",
    )

    def hybrid(new, old, frac_old_=frac_old):
        """
        Generate the hybrid density matrix.
        """
        return new * (1 - frac_old_) + old * frac_old_

    rho_cc_half0 = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_cc[0]) + 1e-14
    rho_cc_half1 = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_cc[1]) + 1e-14
    rho_cc = np.array([rho_cc_half0, rho_cc_half1])
    print(np.sum(rho_cc_half0 * weights), np.sum(rho_cc_half1 * weights))
    if np.sum(rho_cc_half0 * weights) < 1e-7:
        self.spin_list = [1]
        print("Warning: rho_cc_half0 is zero.")
    elif np.sum(rho_cc_half1 * weights) < 1e-7:
        self.spin_list = [0]
        print("Warning: rho_cc_half1 is zero.")
    else:
        self.spin_list = [0, 1]

    if load_inv and Path(self.data_save_path / "exc_grids.npy").exists():
        print("Load data from saved_data: exc_grids, exc_over_rho_grids.")
        exc_grids = np.load(self.data_save_path / "exc_grids.npy")
        exc_over_rho_grids = np.load(self.data_save_path / "exc_over_rho_grids.npy")
    else:
        print("Calculating exc_grids")
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        exc_grids = np.zeros_like(rho_cc)
        exc_over_rho_grids = np.zeros_like(rho_cc)
        for i_spin in self.spin_list:
            dm12 = dm2_cc[2 * i_spin] - oe.contract(
                "pq,rs->pqrs",
                dm1_cc[i_spin],
                dm1_cc[0] + dm1_cc[1],
            )
            if i_spin == 0:
                dm12 += dm2_cc[1]
            else:
                dm12 += dm2_cc[1].transpose(2, 3, 0, 1)

            for i_batch, j_batch, k_batch, l_batch in product(
                range(n_batchs),
                range(n_batchs),
                range(n_batchs),
                range(n_batchs),
            ):
                nao_slice_i = (
                    n_slices
                    if i_batch != n_batchs - 1
                    else self.mol.nao - n_slices * i_batch
                )
                nao_slice_j = (
                    n_slices
                    if j_batch != n_batchs - 1
                    else self.mol.nao - n_slices * j_batch
                )
                nao_slice_k = (
                    n_slices
                    if k_batch != n_batchs - 1
                    else self.mol.nao - n_slices * k_batch
                )
                nao_slice_l = (
                    n_slices
                    if l_batch != n_batchs - 1
                    else self.mol.nao - n_slices * l_batch
                )

                i_slice = slice(n_slices * i_batch, n_slices * i_batch + nao_slice_i)
                j_slice = slice(n_slices * j_batch, n_slices * j_batch + nao_slice_j)
                k_slice = slice(n_slices * k_batch, n_slices * k_batch + nao_slice_k)
                l_slice = slice(n_slices * l_batch, n_slices * l_batch + nao_slice_l)

                expr_rinv_dm2_r = oe.contract_expression(
                    "ijkl,i,j,kl->",
                    dm12[i_slice, j_slice, k_slice, l_slice],
                    (nao_slice_i,),
                    (nao_slice_j,),
                    (nao_slice_k, nao_slice_l),
                    constants=[0],
                    optimize="optimal",
                )

                for i, coord in enumerate(tqdm(coords)):
                    ao_0_i = ao_value[0][i]
                    with self.mol.with_rinv_origin(coord):
                        rinv = self.mol.intor("int1e_rinv")
                        exc_grids[i_spin, i] += 0.5 * expr_rinv_dm2_r(
                            ao_0_i[i_slice],
                            ao_0_i[j_slice],
                            rinv[k_slice, l_slice],
                            backend="torch",
                        )

                del expr_rinv_dm2_r
                gc.collect()
                torch.cuda.empty_cache()

            # print(f"After 2Rdm,\n {torch.cuda.memory_summary()}.\n")
            # eri = self.mol.intor("int2e")
            # ene_vc = np.sum(exc_grids[i_spin] * weights)
            # ene_cc_ele = 0.5 * np.einsum("pqrs,pqrs", eri, dm12)
            # print(f"Error: {(1e3 * (ene_vc - ene_cc_ele)):.5f} mHa")
            del dm12
            gc.collect()

            exc_over_rho_grids[i_spin] = exc_grids[i_spin] / rho_cc[i_spin]

        del dm2_cc
        gc.collect()

        np.save(self.data_save_path / "exc_grids.npy", exc_grids)
        np.save(self.data_save_path / "exc_over_rho_grids.npy", exc_over_rho_grids)

    # if False:
    if load_inv and Path(self.data_save_path / "emax.npy").exists():
        print("Load data from saved_data: emax, taup_rho_wf, tau_rho_wf, v_vxc_e_taup.")
        emax = np.load(self.data_save_path / "emax.npy")
        taup_rho_wf = np.load(self.data_save_path / "taup_rho_wf.npy")
        tau_rho_wf = np.load(self.data_save_path / "tau_rho_wf.npy")
        v_vxc_e_taup = np.load(self.data_save_path / "v_vxc_e_taup.npy")
    else:
        mo_a = mf.mo_coeff[0]
        mo_b = mf.mo_coeff[1]
        nmoa = mo_a.shape[1]
        nmob = mo_b.shape[1]
        print(f"nmoa: {nmoa}, nmob: {nmob}")

        dm1_cc_mo = mycc.make_rdm1()
        dm2_cc_mo = mycc.make_rdm2()
        eriaa = pyscf.ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa] * 4)
        eribb = pyscf.ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob] * 4)
        eriab = pyscf.ao2mo.kernel(mf._eri, (mo_a, mo_a, mo_b, mo_b), compact=False)
        eriab = eriab.reshape([nmoa, nmoa, nmob, nmob])
        eri_mo = np.array([eriaa, eriab, eribb])

        e_bar_r_wf = np.zeros((2, len(coords)))

        for i_spin in self.spin_list:
            h1_mo = np.einsum("ab,ai,bj->ij", h1e, mo[i_spin], mo[i_spin])
            generalized_fock = dm1_cc_mo[i_spin] @ h1_mo
            if i_spin == 0:
                generalized_fock += oe.contract(
                    "aijk,bijk->ba",
                    dm2_cc_mo[2 * i_spin],
                    eri_mo[2 * i_spin],
                )
                generalized_fock += oe.contract(
                    "aijk,bijk->ba",
                    dm2_cc_mo[1],
                    eri_mo[1],
                )
            else:
                generalized_fock += oe.contract(
                    "jkai,jkbi->ba",
                    dm2_cc_mo[2 * i_spin],
                    eri_mo[2 * i_spin],
                )
                generalized_fock += oe.contract(
                    "jkai,jkbi->ba",
                    dm2_cc_mo[1],
                    eri_mo[1],
                )

            # for a_batch, b_batch, i_batch, j_batch, k_batch in product(
            #     range(n_batchs),
            #     range(n_batchs),
            #     range(n_batchs),
            #     range(n_batchs),
            #     range(n_batchs),
            # ):
            #     print(f"Batch: {a_batch, b_batch, i_batch, j_batch, k_batch}")
            #     nao_slice_a = (
            #         n_slices
            #         if a_batch != n_batchs - 1
            #         else self.mol.nao - n_slices * a_batch
            #     )
            #     nao_slice_b = (
            #         n_slices
            #         if b_batch != n_batchs - 1
            #         else self.mol.nao - n_slices * b_batch
            #     )
            #     nao_slice_i = (
            #         n_slices
            #         if i_batch != n_batchs - 1
            #         else self.mol.nao - n_slices * i_batch
            #     )
            #     nao_slice_j = (
            #         n_slices
            #         if j_batch != n_batchs - 1
            #         else self.mol.nao - n_slices * j_batch
            #     )
            #     nao_slice_k = (
            #         n_slices
            #         if k_batch != n_batchs - 1
            #         else self.mol.nao - n_slices * k_batch
            #     )

            #     a_slice = slice(n_slices * a_batch, n_slices * a_batch + nao_slice_a)
            #     b_slice = slice(n_slices * b_batch, n_slices * b_batch + nao_slice_b)
            #     i_slice = slice(n_slices * i_batch, n_slices * i_batch + nao_slice_i)
            #     j_slice = slice(n_slices * j_batch, n_slices * j_batch + nao_slice_j)
            #     k_slice = slice(n_slices * k_batch, n_slices * k_batch + nao_slice_k)

            #     expr_dm2_cc = oe.contract_expression(
            #         "aijk,bijk->ba",
            #         (nao_slice_a, nao_slice_i, nao_slice_j, nao_slice_k),
            #         (nao_slice_b, nao_slice_i, nao_slice_j, nao_slice_k),
            #         optimize="optimal",
            #     )
            #     generalized_fock[b_slice, a_slice] += expr_dm2_cc(
            #         dm2_cc_mo[1][a_slice, i_slice, j_slice, k_slice],
            #         eri_mo[2 * i_spin + 1][b_slice, i_slice, j_slice, k_slice],
            #         backend="torch",
            #     )
            #     # generalized_fock[b_slice, a_slice] += expr_dm2_cc(
            #     #     dm2_cc_mo[2 * i_spin][a_slice, i_slice, j_slice, k_slice],
            #     #     eri_mo[2 * i_spin][b_slice, i_slice, j_slice, k_slice],
            #     #     backend="torch",
            #     # )

            #     del expr_dm2_cc
            #     gc.collect()
            #     torch.cuda.empty_cache()

            generalized_fock = 0.5 * (generalized_fock + generalized_fock.T)
            eig_e, eig_v = np.linalg.eigh(generalized_fock)
            eig_v = mo_cc[i_spin] @ eig_v

            expr_e_bar_r_wf = oe.contract_expression(
                "i,mi,ni,pm,pn->p",
                (self.mol.nao,),
                eig_v,
                eig_v,
                ao_0,
                ao_0,
                constants=[1, 2, 3, 4],
                optimize="optimal",
            )

            e_bar_r_wf[i_spin] = (
                expr_e_bar_r_wf(eig_e, backend="torch") / rho_cc[i_spin]
            )

        del expr_e_bar_r_wf, dm2_cc_mo, eri_mo
        gc.collect()
        torch.cuda.empty_cache()

        emax = np.zeros(2)
        taup_rho_wf = np.zeros((2, len(coords)))
        tau_rho_wf = np.zeros((2, len(coords)))
        v_vxc_e_taup = np.zeros((2, len(coords)))

        for i_spin in self.spin_list:
            eigs_e_dm1, eigs_v_dm1 = LA.eigh(dm1_cc_mo[i_spin])
            eigs_v_dm1 = mo_cc[i_spin] @ eigs_v_dm1

            taup_rho_wf[i_spin] = gen_taup_rho(
                rho_cc[i_spin],
                eigs_v_dm1,
                eigs_e_dm1,
                oe_taup_rho,
                backend="torch",
            )

            tau_rho_wf[i_spin] = gen_tau_rho(
                rho_cc[i_spin],
                eigs_v_dm1,
                eigs_e_dm1,
                oe_tau_rho,
                backend="torch",
            )

            emax[i_spin] = np.max(e_bar_r_wf[i_spin])
            v_vxc_e_taup[i_spin] = (
                exc_over_rho_grids[i_spin] * 2
                + taup_rho_wf[i_spin] / rho_cc[i_spin]
                - e_bar_r_wf[i_spin]
            )

        print(f"After prepare,\n {torch.cuda.memory_summary()}.\n")

        np.save(self.data_save_path / "emax.npy", emax)
        np.save(self.data_save_path / "taup_rho_wf.npy", taup_rho_wf)
        np.save(self.data_save_path / "tau_rho_wf.npy", tau_rho_wf)
        np.save(self.data_save_path / "v_vxc_e_taup.npy", v_vxc_e_taup)

    print(
        f"int1e_grids will consume about {len(coords) * self.mol.nao**2 * 8 / 1024**3:.2f} GB memory on cpu."
    )
    int1e_grids = self.mol.intor("int1e_grids", grids=coords)

    if load_inv and Path(self.data_save_path / "dm1_inv.npy").exists():
        print("Load data from saved_data: dm1_inv, vxc_inv, tau_rho_ks, taup_rho_ks.")
        dm1_inv = np.load(self.data_save_path / "dm1_inv.npy")
        vxc_inv = np.load(self.data_save_path / "vxc_inv.npy")
        tau_rho_ks = np.load(self.data_save_path / "tau_rho_ks.npy")
        taup_rho_ks = np.load(self.data_save_path / "taup_rho_ks.npy")
    else:
        eigvecs_inv = mf.mo_energy.copy()
        mo_inv = mf.mo_coeff.copy()
        dm1_inv = mf.make_rdm1(ao_repr=True)
        vxc_inv = np.array(
            [
                pyscf.dft.libxc.eval_xc(
                    "lda",
                    pyscf.dft.numint.eval_rho(
                        self.mol,
                        ao_0,
                        dm1_inv[0] * 2,
                    ),
                )[1][0],
                pyscf.dft.libxc.eval_xc(
                    "lda",
                    pyscf.dft.numint.eval_rho(
                        self.mol,
                        ao_0,
                        dm1_inv[1] * 2,
                    ),
                )[1][0],
            ]
        )

        oe_ebar_r_ks = [
            oe.contract_expression(
                "i,mi,ni,pm,pn->p",
                (nocc[0],),
                (nao, nocc[0]),
                (nao, nocc[0]),
                ao_0,
                ao_0,
                constants=[3, 4],
                optimize="optimal",
            ),
            oe.contract_expression(
                "i,mi,ni,pm,pn->p",
                (nocc[1],),
                (nao, nocc[1]),
                (nao, nocc[1]),
                ao_0,
                ao_0,
                constants=[3, 4],
                optimize="optimal",
            ),
        ]

        oe_fock = oe.contract_expression(
            "p,p,pa,pb->ab",
            np.shape(ao_0[:, 0]),
            np.shape(ao_0[:, 0]),
            ao_0,
            ao_0,
            constants=[2, 3],
            optimize="optimal",
        )

        diis = (DIIS(self.mol.nao, n=diis_n), DIIS(self.mol.nao, n=diis_n))

        for i in range(250):
            dm1_inv_r = np.array(
                [
                    pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv[0]) + 1e-14,
                    pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv[1]) + 1e-14,
                ]
            )

            potential_shift = np.zeros(2)
            for i_spin in self.spin_list:
                potential_shift[i_spin] = emax[i_spin] - np.max(
                    eigvecs_inv[i_spin][: nocc[i_spin]]
                )

            ebar_ks = np.array(
                [
                    oe_ebar_r_ks[0](
                        eigvecs_inv[0][: nocc[0]] + potential_shift[[0]],
                        mo_inv[0][:, : nocc[0]],
                        mo_inv[0][:, : nocc[0]],
                        backend="torch",
                    )
                    / dm1_inv_r[0],
                    oe_ebar_r_ks[1](
                        eigvecs_inv[1][: nocc[1]] + potential_shift[[1]],
                        mo_inv[1][:, : nocc[1]],
                        mo_inv[1][:, : nocc[1]],
                        backend="torch",
                    )
                    / dm1_inv_r[1],
                ]
            )

            taup_rho_ks = np.array(
                [
                    gen_taup_rho(
                        dm1_inv_r[0],
                        mo_inv[0][:, : nocc[0]],
                        np.ones(nocc[0]),
                        oe_taup_rho,
                        backend="torch",
                    ),
                    gen_taup_rho(
                        dm1_inv_r[1],
                        mo_inv[1][:, : nocc[1]],
                        np.ones(nocc[1]),
                        oe_taup_rho,
                        backend="torch",
                    ),
                ]
            )

            vxc_inv_old = vxc_inv.copy()
            vxc_inv = np.array(
                [
                    v_vxc_e_taup[0] + ebar_ks[0] - taup_rho_ks[0] / dm1_inv_r[0],
                    v_vxc_e_taup[1] + ebar_ks[1] - taup_rho_ks[1] / dm1_inv_r[1],
                ]
            )

            for i_batch_grids in range(n_batchs_grids):
                ngrids_slice_i = (
                    n_slice_grids
                    if i_batch_grids != n_slice_grids - 1
                    else len(coords) - n_slice_grids * i_batch_grids
                )
                i_slice_grids = slice(
                    n_slice_grids * i_batch_grids,
                    n_slice_grids * i_batch_grids + ngrids_slice_i,
                )
                exp_int1e_grids = oe.contract_expression(
                    "pij,ij->p",
                    int1e_grids[i_slice_grids, :, :],
                    (nao, nao),
                    constants=[0],
                    optimize="optimal",
                )

                tmp_grids = exp_int1e_grids(
                    (dm1_cc[0] + dm1_cc[1] - dm1_inv[0] - dm1_inv[1]),
                    backend="torch",
                )
                for i_spin in self.spin_list:
                    vxc_inv[i_spin, i_slice_grids] -= tmp_grids
                del exp_int1e_grids
                gc.collect()
                torch.cuda.empty_cache()

            error_vxc = np.linalg.norm((vxc_inv - vxc_inv_old) * weights)
            vxc_inv = hybrid(vxc_inv, vxc_inv_old)
            vj_inv = mf.get_jk(self.mol, dm1_inv[0] + dm1_inv[1], 1)[0]
            xc_v = np.array(
                [
                    h1e + vj_inv + oe_fock(vxc_inv[0], weights, backend="torch"),
                    h1e + vj_inv + oe_fock(vxc_inv[1], weights, backend="torch"),
                ]
            )

            mat_fock = np.zeros((2, self.mol.nao, self.mol.nao))
            dm1_inv_old = dm1_inv.copy()

            for i_spin in self.spin_list:
                diis[i_spin].add(
                    xc_v[i_spin],
                    mat_s @ dm1_inv_old[i_spin] @ xc_v[i_spin]
                    - xc_v[i_spin] @ dm1_inv_old[i_spin] @ mat_s,
                )
                mat_fock[i_spin] = diis[i_spin].hybrid()
                eigvecs_inv[i_spin], mo_inv[i_spin] = np.linalg.eigh(
                    mat_hs @ mat_fock[i_spin] @ mat_hs
                )
                mo_inv[i_spin] = mat_hs @ mo_inv[i_spin]

                dm1_inv[i_spin] = (
                    mo_inv[i_spin][:, : nocc[i_spin]]
                    @ mo_inv[i_spin][:, : nocc[i_spin]].T
                )
            error_dm1 = np.linalg.norm(dm1_inv - dm1_inv_old)

            print(
                f"step:{i:<8}",
                f"error of vxc: {error_vxc::<10.5e}",
                f"dm: {error_dm1::<10.5e}",
                f"shift: {np.array2string(potential_shift, formatter={'float_kind': lambda x: f'{x:<7.2e}'})}",
                f"emax: {np.array2string(emax, formatter={'float_kind': lambda x: f'{x:<7.2e}'})}",
                flush=True,
            )
            if (i > 0) and (error_vxc < 1e-8):
                break

        dm1_inv_r = np.array(
            [
                pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv[0]) + 1e-14,
                pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv[1]) + 1e-14,
            ]
        )
        tau_rho_ks = np.array(
            [
                gen_tau_rho(
                    dm1_inv_r[0],
                    mo_inv[0][:, : nocc[0]],
                    np.ones(nocc[0]),
                    oe_tau_rho,
                    backend="torch",
                ),
                gen_tau_rho(
                    dm1_inv_r[1],
                    mo_inv[1][:, : nocc[1]],
                    np.ones(nocc[1]),
                    oe_tau_rho,
                    backend="torch",
                ),
            ]
        )

        del oe_tau_rho, oe_ebar_r_ks, oe_fock
        gc.collect()
        torch.cuda.empty_cache()

        # print(f"After inv,\n {torch.cuda.memory_summary()}.\n")

        np.save(self.data_save_path / "dm1_inv.npy", dm1_inv)
        np.save(self.data_save_path / "vxc_inv.npy", vxc_inv)
        np.save(self.data_save_path / "tau_rho_ks.npy", tau_rho_ks)
        np.save(self.data_save_path / "taup_rho_ks.npy", taup_rho_ks)

    kin_correct = np.sum((tau_rho_wf - tau_rho_ks) * weights)
    kin_correct1 = np.sum((taup_rho_wf - taup_rho_ks) * weights)

    rho_inv = np.array(
        [
            pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv[0]) + 1e-14,
            pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv[1]) + 1e-14,
        ]
    )
    rho_dft = np.array(
        [
            pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_dft[0]) + 1e-14,
            pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_dft[1]) + 1e-14,
        ]
    )
    exc_grids_fake = exc_grids.copy()
    exc_grids_fake1 = exc_grids.copy()
    print("rho_inv: ", np.sum(rho_inv[0] * weights), np.sum(rho_inv[1] * weights))
    print("rho_dft: ", np.sum(rho_dft[0] * weights), np.sum(rho_dft[1] * weights))
    print("rho_cc: ", np.sum(rho_cc[0] * weights), np.sum(rho_cc[1] * weights))

    for i_batch_grids in range(n_batchs_grids):
        ngrids_slice_i = (
            n_slice_grids
            if i_batch_grids != n_slice_grids - 1
            else len(coords) - n_slice_grids * i_batch_grids
        )
        i_slice_grids = slice(
            n_slice_grids * i_batch_grids,
            n_slice_grids * i_batch_grids + ngrids_slice_i,
        )
        for i_spin in self.spin_list:
            vele = np.einsum(
                "pij,ij->p",
                int1e_grids[i_slice_grids, :, :],
                dm1_cc[0] + dm1_cc[1],
            )
            diif_rho_grids = (
                rho_cc[i_spin][i_slice_grids] - rho_inv[i_spin][i_slice_grids]
            )
            exc_grids_fake[i_spin, i_slice_grids] += vele * diif_rho_grids
            exc_grids_fake1[i_spin, i_slice_grids] += vele * diif_rho_grids

    for i_spin in self.spin_list:
        for i, coord in enumerate(tqdm(coords)):
            for i_atom in range(self.mol.natm):
                distance = np.linalg.norm(self.mol.atom_coords()[i_atom] - coord)
                if distance > 1e-3:
                    exc_grids_fake[i_spin, i] -= (
                        (rho_cc[i_spin, i] - rho_inv[i_spin, i])
                        * self.mol.atom_charges()[i_atom]
                        / distance
                    )
                else:
                    exc_grids_fake[i_spin, i] -= (
                        (rho_cc[i_spin, i] - rho_inv[i_spin, i])
                        * self.mol.atom_charges()[i_atom]
                        / 1e-3
                    )

                if distance > 1e-2:
                    exc_grids_fake1[i_spin, i] -= (
                        (rho_cc[i_spin, i] - rho_inv[i_spin, i])
                        * self.mol.atom_charges()[i_atom]
                        / distance
                    )
                else:
                    exc_grids_fake1[i_spin, i] -= (
                        (rho_cc[i_spin, i] - rho_inv[i_spin, i])
                        * self.mol.atom_charges()[i_atom]
                        / 1e-2
                    )

    exc_over_rho_grids_fake = np.array(
        [exc_grids_fake[0] / rho_inv[0], exc_grids_fake[1] / rho_inv[1]]
    )
    exc_over_rho_grids_fake1 = np.array(
        [exc_grids_fake1[0] / rho_inv[0], exc_grids_fake1[1] / rho_inv[1]]
    )

    save_data = {}
    save_data["energy"] = AU2KJMOL * e_cc
    save_data["correct kinetic energy"] = AU2KJMOL * kin_correct
    save_data["correct kinetic energy1"] = AU2KJMOL * kin_correct1
    save_data["energy_dft"] = AU2KJMOL * (mdft.e_tot - e_cc)

    print(
        AU2KJMOL
        * (
            np.sum(self.mol.intor("int1e_kin") * (dm1_inv[0] + dm1_inv[1]))
            - np.sum(self.mol.intor("int1e_kin") * (dm1_cc[0] + dm1_cc[1]))
        )
    )
    print(
        AU2KJMOL
        * (
            np.sum(self.mol.intor("int1e_nuc") * (dm1_inv[0] + dm1_inv[1]))
            - np.sum(self.mol.intor("int1e_nuc") * (dm1_cc[0] + dm1_cc[1]))
        )
    )
    print(
        AU2KJMOL
        * (
            np.sum(
                mf.get_jk(self.mol, dm1_inv[0] + dm1_inv[1], 1)[0]
                * (dm1_inv[0] + dm1_inv[1])
            )
            - np.sum(
                mf.get_jk(self.mol, dm1_cc[0] + dm1_cc[1], 1)[0]
                * (dm1_cc[0] + dm1_cc[1])
            )
        )
    )
    hcore_vj_energy = (
        np.sum(h1e * (dm1_inv[0] + dm1_inv[1]))
        + 0.5
        * np.sum(
            mf.get_jk(self.mol, dm1_inv[0] + dm1_inv[1], 1)[0]
            * (dm1_inv[0] + dm1_inv[1])
        )
        + self.mol.energy_nuc()
    )

    save_data["energy_inv"] = AU2KJMOL * (
        (
            hcore_vj_energy
            + np.sum(exc_over_rho_grids_fake * rho_inv * weights)
            + kin_correct
        )
        - e_cc
    )
    save_data["energy_inv1"] = AU2KJMOL * (
        (
            hcore_vj_energy
            + np.sum(exc_over_rho_grids_fake1 * rho_inv * weights)
            + kin_correct
        )
        - e_cc
    )
    save_data["energy_inv_real"] = AU2KJMOL * (
        (hcore_vj_energy + np.sum(exc_over_rho_grids * rho_inv * weights) + kin_correct)
        - e_cc
    )
    save_data["energy_cc_real"] = AU2KJMOL * (
        (
            np.sum(h1e * (dm1_cc[0] + dm1_cc[1]))
            + 0.5
            * np.sum(
                mf.get_jk(self.mol, dm1_cc[0] + dm1_cc[1], 1)[0]
                * (dm1_cc[0] + dm1_cc[1])
            )
            + self.mol.energy_nuc()
            + np.sum(exc_over_rho_grids * rho_cc * weights)
        )
        - e_cc
    )

    error_inv_r = np.sum(np.abs(rho_inv - rho_cc) * weights)
    error_dft_r = np.sum(np.abs(rho_dft - rho_cc) * weights)
    save_data["error of dm1_inv"] = error_inv_r
    save_data["error of dm1_dft"] = error_dft_r

    dipole_x_core = 0
    for i_atom in range(self.mol.natm):
        dipole_x_core += (
            self.mol.atom_charges()[i_atom] * self.mol.atom_coords()[i_atom][0]
        )
    dipole_x = dipole_x_core - np.sum(rho_cc * coords[:, 0] * weights)
    dipole_x_inv = dipole_x_core - np.sum(rho_inv * coords[:, 0] * weights)
    dipole_x_dft = dipole_x_core - np.sum(rho_dft * coords[:, 0] * weights)
    save_data["dipole_x"] = dipole_x
    save_data["dipole_x_inv"] = dipole_x_inv
    save_data["dipole_x_dft"] = dipole_x_dft

    dipole_y_core = 0
    for i_atom in range(self.mol.natm):
        dipole_y_core += (
            self.mol.atom_charges()[i_atom] * self.mol.atom_coords()[i_atom][1]
        )
    dipole_y = dipole_y_core - np.sum(rho_cc * coords[:, 1] * weights)
    dipole_y_inv = dipole_y_core - np.sum(rho_inv * coords[:, 1] * weights)
    dipole_y_dft = dipole_y_core - np.sum(rho_dft * coords[:, 1] * weights)
    save_data["dipole_y"] = dipole_y
    save_data["dipole_y_inv"] = dipole_y_inv
    save_data["dipole_y_dft"] = dipole_y_dft

    dipole_z_core = 0
    for i_atom in range(self.mol.natm):
        dipole_z_core += (
            self.mol.atom_charges()[i_atom] * self.mol.atom_coords()[i_atom][2]
        )
    dipole_z = dipole_z_core - np.sum(rho_cc * coords[:, 2] * weights)
    dipole_z_inv = dipole_z_core - np.sum(rho_inv * coords[:, 2] * weights)
    dipole_z_dft = dipole_z_core - np.sum(rho_dft * coords[:, 2] * weights)
    save_data["dipole_z"] = dipole_z
    save_data["dipole_z_inv"] = dipole_z_inv
    save_data["dipole_z_dft"] = dipole_z_dft

    with open(DATA_PATH / f"save_data_{self.name}.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4)

    for i_spin in range(2):
        ao_value = pyscf.dft.numint.eval_ao(self.mol, coords, deriv=1)
        inv_r_3 = pyscf.dft.numint.eval_rho(
            self.mol, ao_value, dm1_inv[i_spin] * 2, xctype="GGA"
        )
        evxc_b3lyp = pyscf.dft.libxc.eval_xc("b3lyp", inv_r_3)
        exc_b3lyp = evxc_b3lyp[0]
        vxc_b3lyp = evxc_b3lyp[1][0]
        data_grids_norm = process_input(inv_r_3, grids)

        print(
            f"spin {i_spin}, number of ele: {np.sum(grids.vector_to_matrix(rho_cc[i_spin]) * grids.vector_to_matrix(weights))}"
        )

        # Note the input is rho_inv_4_norm, which satisfies the original DFT routine,
        # \rho = eval_rho(2 * dm1) not \rho = eval_rho(dm1) will used in eval_xc(\rho) function in the pyscf.dft.numint.
        # So, rho_inv_4_norm[0] = 2 * rho_inv[i_spin] is used in the data saving process.
        np.savez_compressed(
            DATA_PATH / f"data_{self.name}_{i_spin}.npz",
            dm_cc=dm1_cc[i_spin],
            dm_inv=dm1_inv[i_spin],
            rho_cc=grids.vector_to_matrix(rho_cc[i_spin]),
            rho_inv=grids.vector_to_matrix(rho_inv[i_spin]),
            weights=grids.vector_to_matrix(weights),
            vxc=grids.vector_to_matrix(vxc_inv[i_spin]),
            vxc_b3lyp=grids.vector_to_matrix(vxc_inv[i_spin] - vxc_b3lyp),
            vxc1_b3lyp=grids.vector_to_matrix(vxc_inv[i_spin] - evxc_b3lyp[0]),
            exc=grids.vector_to_matrix(exc_over_rho_grids_fake[i_spin]),
            exc_real=grids.vector_to_matrix(exc_over_rho_grids[i_spin]),
            exc_tr=grids.vector_to_matrix(
                exc_over_rho_grids_fake[i_spin]
                + (tau_rho_wf[i_spin] - tau_rho_ks[i_spin]) / rho_inv[i_spin]
            ),
            exc1_tr=grids.vector_to_matrix(
                exc_over_rho_grids_fake1[i_spin]
                + (tau_rho_wf[i_spin] - tau_rho_ks[i_spin]) / rho_inv[i_spin]
            ),
            exc_tr_real=grids.vector_to_matrix(
                exc_over_rho_grids[i_spin]
                + (tau_rho_wf[i_spin] - tau_rho_ks[i_spin]) / rho_inv[i_spin]
            ),
            exc1_tr_real=grids.vector_to_matrix(
                exc_over_rho_grids[i_spin]
                + (tau_rho_wf[i_spin] - tau_rho_ks[i_spin]) / rho_inv[i_spin]
            ),
            exc_tr_b3lyp=grids.vector_to_matrix(
                exc_over_rho_grids_fake[i_spin]
                + (tau_rho_wf[i_spin] - tau_rho_ks[i_spin]) / rho_inv[i_spin]
                - exc_b3lyp
            ),
            exc1_tr_b3lyp=grids.vector_to_matrix(
                exc_over_rho_grids_fake1[i_spin]
                + (tau_rho_wf[i_spin] - tau_rho_ks[i_spin]) / rho_inv[i_spin]
                - exc_b3lyp
            ),
            rho_inv_4_norm=data_grids_norm,
            coords_x=grids.vector_to_matrix(coords[:, 0]),
            coords_y=grids.vector_to_matrix(coords[:, 1]),
            coords_z=grids.vector_to_matrix(coords[:, 2]),
        )
