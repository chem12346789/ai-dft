import json
import gc
from pathlib import Path
from itertools import product

import numpy as np
from tqdm import tqdm
import torch
import pyscf
import scipy.linalg as LA
import opt_einsum as oe

from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import ccsd_t_slow as ccsd_t
from pyscf.cc import ccsd_rdm
from pyscf.cc.ccsd_t_rdm_slow import _gamma1_intermediates
from pyscf.cc.ccsd_t_rdm_slow import _gamma2_intermediates

from cadft.utils.gen_tau import gen_taup_rho, gen_taul_rho, gen_tau_rho
from cadft.utils.Grids import Grid
from cadft.utils.env_var import DATA_PATH, DATA_SAVE_PATH
from cadft.utils.diis import DIIS
from cadft.utils.DataBase import process_input

AU2KJMOL = 2625.5
CCSDT = False


def mrks_diis(self, frac_old, load_inv=True, diis_n=15):
    """
    Generate 1-RDM.
    """
    self.data_save_path = DATA_SAVE_PATH / f"{self.name}"
    Path(self.data_save_path).mkdir(parents=True, exist_ok=True)
    n_slices = 150

    mdft = pyscf.scf.RKS(self.mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    mf = pyscf.scf.RHF(self.mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.direct = True
    mycc.incore_complete = True
    mycc.async_io = False

    _, t1, t2 = mycc.kernel()
    if CCSDT:
        eris = mycc.ao2mo()
        e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
        e_cc = mycc.e_tot + e3ref
        print(f"CCSD(T) correlation energy: {e3ref:.8f}")
        l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
        dm1_cc = ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris, ao_repr=True)
    else:
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        e_cc = mycc.e_tot

    h1e = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
    mo = mf.mo_coeff
    nocc = self.mol.nelec[0]
    norb = mo.shape[1]
    print(h1e.shape)

    n_batchs = self.mol.nao // n_slices + 1

    mat_s = self.mol.intor("int1e_ovlp")
    mat_hs = LA.fractional_matrix_power(mat_s, -0.5).real

    grids = Grid(self.mol, level=1)
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
        (norb,),
        (norb,),
        ao_1,
        constants=[0, 3],
        optimize="optimal",
    )

    oe_taul_rho = oe.contract_expression(
        "n,kpn->pk",
        (norb,),
        ao_1,
        constants=[1],
        optimize="optimal",
    )

    oe_tau_rho = oe.contract_expression(
        "pm,m,n,pn->p",
        ao_0,
        (norb,),
        (norb,),
        ao_2_diag,
        constants=[0, 3],
        optimize="optimal",
    )

    def hybrid(new, old, frac_old_=frac_old):
        """
        Generate the hybrid density matrix.
        """
        return new * (1 - frac_old_) + old * frac_old_

    rho_cc = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_cc) + 1e-14
    exc_grids = np.zeros_like(rho_cc)

    if load_inv and Path(self.data_save_path / "exc_grids.npy").exists():
        print("Load data from saved_data: exc_grids, exc_over_rho_grids.")
        exc_grids = np.load(self.data_save_path / "exc_grids.npy")
        exc_over_rho_grids = np.load(self.data_save_path / "exc_over_rho_grids.npy")
    else:
        print("Calculating exc_grids")
        if CCSDT:
            d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
            d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
            dm2_cc = ccsd_rdm._make_rdm2(mycc, d1, d2, True, True, ao_repr=True)
            dm12 = dm2_cc - oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)
            del dm2_cc, d1, d2
        else:
            dm2_cc = mycc.make_rdm2(ao_repr=True)
            dm12 = dm2_cc - oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)
            del dm2_cc
        gc.collect()

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
                    exc_grids[i] += 0.5 * expr_rinv_dm2_r(
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                        backend="torch",
                    )

            del expr_rinv_dm2_r
            gc.collect()
            torch.cuda.empty_cache()

        exc_over_rho_grids = exc_grids / rho_cc
        # print(f"After 2Rdm,\n {torch.cuda.memory_summary()}.\n")

        # ene_vc = np.sum(exc_over_rho_grids * rho_cc * weights)
        # print(f"Error: {(1e3 * (ene_vc - ene_cc_ele)):.5f} mHa")
        # eri = self.mol.intor("int2e")
        # ene_cc_ele = (
        #     np.einsum("pq,pq", h1e, dm1_cc)
        #     + 0.5 * np.einsum("pqrs,pqrs", eri, dm2_cc)
        #     + self.mol.energy_nuc()
        # )
        # print(f"Error: {(ene_cc_ele- e_cc):.5f} mHa")

        del dm12
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
        if CCSDT:
            dm1_cc_mo = ccsd_t_rdm.make_rdm1(
                mycc, t1, t2, l1, l2, eris=eris, ao_repr=False
            )
            dm2_cc_mo = ccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris=eris)
        else:
            dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
            dm2_cc_mo = mycc.make_rdm2(ao_repr=False)

        h1_mo = np.einsum("ab,ai,bj->ij", h1e, mo, mo)
        eri = self.mol.intor("int2e")
        eri_mo = pyscf.ao2mo.kernel(eri, mo, compact=False)

        generalized_fock = dm1_cc_mo @ h1_mo
        for a_batch, b_batch, i_batch, j_batch, k_batch in product(
            range(n_batchs),
            range(n_batchs),
            range(n_batchs),
            range(n_batchs),
            range(n_batchs),
        ):
            print(f"Batch: {a_batch, b_batch, i_batch, j_batch, k_batch}")
            nao_slice_a = (
                n_slices
                if a_batch != n_batchs - 1
                else self.mol.nao - n_slices * a_batch
            )
            nao_slice_b = (
                n_slices
                if b_batch != n_batchs - 1
                else self.mol.nao - n_slices * b_batch
            )
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

            a_slice = slice(n_slices * a_batch, n_slices * a_batch + nao_slice_a)
            b_slice = slice(n_slices * b_batch, n_slices * b_batch + nao_slice_b)
            i_slice = slice(n_slices * i_batch, n_slices * i_batch + nao_slice_i)
            j_slice = slice(n_slices * j_batch, n_slices * j_batch + nao_slice_j)
            k_slice = slice(n_slices * k_batch, n_slices * k_batch + nao_slice_k)

            expr_dm2_cc = oe.contract_expression(
                "aijk,bijk->ba",
                (nao_slice_a, nao_slice_i, nao_slice_j, nao_slice_k),
                (nao_slice_b, nao_slice_i, nao_slice_j, nao_slice_k),
                optimize="optimal",
            )
            generalized_fock[b_slice, a_slice] += expr_dm2_cc(
                dm2_cc_mo[a_slice, i_slice, j_slice, k_slice],
                eri_mo[b_slice, i_slice, j_slice, k_slice],
                backend="torch",
            )

            del expr_dm2_cc
            gc.collect()
            torch.cuda.empty_cache()

        generalized_fock = 0.5 * (generalized_fock + generalized_fock.T)
        eig_e, eig_v = np.linalg.eigh(generalized_fock)
        eig_v = mo @ eig_v
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
        e_bar_r_wf = expr_e_bar_r_wf(eig_e, backend="torch") / rho_cc

        del expr_e_bar_r_wf, dm2_cc_mo, eri, eri_mo
        gc.collect()
        torch.cuda.empty_cache()

        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
        eigs_v_dm1 = mo @ eigs_v_dm1

        taup_rho_wf = gen_taup_rho(
            rho_cc,
            eigs_v_dm1,
            eigs_e_dm1,
            oe_taup_rho,
            backend="torch",
        )

        tau_rho_wf = gen_tau_rho(
            rho_cc,
            eigs_v_dm1,
            eigs_e_dm1,
            oe_tau_rho,
            backend="torch",
        )

        emax = np.max(e_bar_r_wf)
        v_vxc_e_taup = exc_over_rho_grids * 2 + taup_rho_wf / rho_cc - e_bar_r_wf

        # print(f"After prepare,\n {torch.cuda.memory_summary()}.\n")
        print(f"v_vxc_e_taup: {np.linalg.norm(v_vxc_e_taup):>.5f}")
        print(f"exc_over_rho_grids: {np.linalg.norm(exc_over_rho_grids):>.5f}")
        print(f"taup_rho_wf: {np.linalg.norm(taup_rho_wf):>.5f}")
        print(f"rho_cc: {np.linalg.norm(rho_cc):>.5f}")
        print(f"e_bar_r_wf: {np.linalg.norm(e_bar_r_wf):>.5f}")

        np.save(self.data_save_path / "emax.npy", emax)
        np.save(self.data_save_path / "taup_rho_wf.npy", taup_rho_wf)
        np.save(self.data_save_path / "tau_rho_wf.npy", tau_rho_wf)
        np.save(self.data_save_path / "v_vxc_e_taup.npy", v_vxc_e_taup)

    print(
        f"int1e_grids will consume about {len(coords) * self.mol.nao**2 * 8 / 1024**3:.2f} GB memory on cpu."
    )
    int1e_grids = self.mol.intor("int1e_grids", grids=coords)
    print(int1e_grids.shape)

    # if False:
    if (
        load_inv
        and Path(
            self.data_save_path / "dm1_inv.npy"
        ).exists()
    ):
        print("Load data from saved_data: dm1_inv, vxc_inv, tau_rho_ks, taup_rho_ks.")
        dm1_inv = np.load(self.data_save_path / "dm1_inv.npy")
        vxc_inv = np.load(self.data_save_path / "vxc_inv.npy")
        tau_rho_ks = np.load(self.data_save_path / "tau_rho_ks.npy")
        taup_rho_ks = np.load(self.data_save_path / "taup_rho_ks.npy")
    else:
        eigvecs_inv = mf.mo_energy.copy()
        mo_inv = mo.copy()
        dm1_inv = mf.make_rdm1(ao_repr=True)
        diis = DIIS(self.mol.nao, n=diis_n)

        vxc_inv = pyscf.dft.libxc.eval_xc(
            "b3lyp",
            pyscf.dft.numint.eval_rho(
                self.mol,
                ao_value[:4, :, :],
                dm1_inv,
                xctype="GGA",
            ),
        )[1][0]

        oe_ebar_r_ks = oe.contract_expression(
            "i,mi,ni,pm,pn->p",
            (nocc,),
            (norb, nocc),
            (norb, nocc),
            ao_0,
            ao_0,
            constants=[3, 4],
            optimize="optimal",
        )

        oe_fock = oe.contract_expression(
            "p,p,pa,pb->ab",
            np.shape(ao_0[:, 0]),
            np.shape(ao_0[:, 0]),
            ao_0,
            ao_0,
            constants=[2, 3],
            optimize="optimal",
        )

        for i in range(250):
            dm1_inv_r = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv) + 1e-14

            potential_shift = emax - np.max(eigvecs_inv[:nocc])
            ebar_ks = (
                oe_ebar_r_ks(
                    2 * (eigvecs_inv[:nocc] + potential_shift),
                    mo_inv[:, :nocc],
                    mo_inv[:, :nocc],
                    backend="torch",
                )
                / dm1_inv_r
            )

            taup_rho_ks = gen_taup_rho(
                dm1_inv_r,
                mo_inv[:, :nocc],
                2 * np.ones(nocc),
                oe_taup_rho,
                backend="torch",
            )

            vxc_inv_old = vxc_inv.copy()
            vxc_inv = v_vxc_e_taup + ebar_ks - taup_rho_ks / dm1_inv_r

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
                    (norb, norb),
                    constants=[0],
                    optimize="optimal",
                )
                vxc_inv[i_slice_grids] -= exp_int1e_grids(
                    dm1_cc - dm1_inv, backend="torch"
                )
                del exp_int1e_grids
                gc.collect()
                torch.cuda.empty_cache()

            error_vxc = np.linalg.norm((vxc_inv - vxc_inv_old) * weights)
            vxc_inv = hybrid(vxc_inv, vxc_inv_old)
            xc_v = oe_fock(vxc_inv, weights, backend="torch")
            vj_inv = mf.get_jk(self.mol, dm1_inv, 1)[0]
            xc_v += h1e + vj_inv
            diis.add(xc_v, mat_s @ dm1_inv @ xc_v - xc_v @ dm1_inv @ mat_s)
            xc_v = diis.hybrid()

            eigvecs_inv, mo_inv = np.linalg.eigh(mat_hs @ xc_v @ mat_hs)
            mo_inv = mat_hs @ mo_inv
            dm1_inv_old = dm1_inv.copy()

            dm1_inv = 2 * mo_inv[:, :nocc] @ mo_inv[:, :nocc].T
            error_dm1 = np.linalg.norm(dm1_inv - dm1_inv_old)

            print(
                f"step:{i:<8}",
                f"error of vxc: {error_vxc::<10.5e}",
                f"dm: {error_dm1::<10.5e}",
                f"shift: {potential_shift::<10.5e}",
                f"emax: {np.array2string(emax, formatter={'float_kind': lambda x: f'{x:<7.2e}'})}",
                flush=True,
            )
            if (i > 0) and (error_vxc < 1e-8):
                break

        tau_rho_ks = gen_tau_rho(
            dm1_inv_r,
            mo_inv[:, :nocc],
            2 * np.ones(nocc),
            oe_tau_rho,
            backend="torch",
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
    inv_r = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv) + 1e-14
    dft_r = pyscf.dft.numint.eval_rho(self.mol, ao_0, mdft.make_rdm1()) + 1e-14
    exc_over_rho_grids_fake = exc_over_rho_grids.copy()
    exc_grids_fake = exc_grids.copy()
    exc_grids_fake1 = exc_grids.copy()

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
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids[i_slice_grids, :, :],
            dm1_cc,
        )
        exc_grids_fake[i_slice_grids] += vele * (
            rho_cc[i_slice_grids] - inv_r[i_slice_grids]
        )
        exc_grids_fake1[i_slice_grids] += vele * (
            rho_cc[i_slice_grids] - inv_r[i_slice_grids]
        )

    for i, coord in enumerate(tqdm(coords)):
        for i_atom in range(self.mol.natm):
            distance = np.linalg.norm(self.mol.atom_coords()[i_atom] - coord)
            if distance > 1e-3:
                exc_grids_fake[i] -= (
                    (rho_cc[i] - inv_r[i]) * self.mol.atom_charges()[i_atom] / distance
                )
            else:
                exc_grids_fake[i] -= (
                    (rho_cc[i] - inv_r[i]) * self.mol.atom_charges()[i_atom] / 1e-3
                )

            if distance > 1e-2:
                exc_grids_fake1[i] -= (
                    (rho_cc[i] - inv_r[i]) * self.mol.atom_charges()[i_atom] / distance
                )
            else:
                exc_grids_fake1[i] -= (
                    (rho_cc[i] - inv_r[i]) * self.mol.atom_charges()[i_atom] / 1e-2
                )

    exc_over_rho_grids_fake = exc_grids_fake / inv_r
    exc_over_rho_grids_fake1 = exc_grids_fake1 / inv_r

    save_data = {}
    save_data["energy"] = AU2KJMOL * e_cc
    save_data["correct kinetic energy"] = AU2KJMOL * kin_correct
    save_data["correct kinetic energy1"] = AU2KJMOL * kin_correct1
    save_data["energy_dft"] = AU2KJMOL * (mdft.e_tot - e_cc)

    hcore_vj_energy = (
        np.sum(h1e * dm1_inv)
        + 0.5 * np.sum(mf.get_jk(self.mol, dm1_inv, 1)[0] * dm1_inv)
        + self.mol.energy_nuc()
    )

    save_data["energy_inv"] = AU2KJMOL * (
        (
            hcore_vj_energy
            + np.sum(exc_over_rho_grids_fake * inv_r * weights)
            + kin_correct
        )
        - e_cc
    )
    save_data["energy_inv1"] = AU2KJMOL * (
        (
            hcore_vj_energy
            + np.sum(exc_over_rho_grids_fake1 * inv_r * weights)
            + kin_correct
        )
        - e_cc
    )
    save_data["energy_inv_real"] = AU2KJMOL * (
        (hcore_vj_energy + np.sum(exc_over_rho_grids * inv_r * weights) + kin_correct)
        - e_cc
    )
    save_data["energy_inv_real1"] = AU2KJMOL * (
        (hcore_vj_energy + np.sum(exc_over_rho_grids * inv_r * weights) + kin_correct1)
        - e_cc
    )

    error_inv_r = np.sum(np.abs(inv_r - rho_cc) * weights)
    error_dft_r = np.sum(np.abs(dft_r - rho_cc) * weights)
    save_data["error of dm1_inv"] = error_inv_r
    save_data["error of dm1_dft"] = error_dft_r

    dipole_x_core = 0
    for i_atom in range(self.mol.natm):
        dipole_x_core += (
            self.mol.atom_charges()[i_atom] * self.mol.atom_coords()[i_atom][0]
        )
    dipole_x = dipole_x_core - np.sum(rho_cc * coords[:, 0] * weights)
    dipole_x_inv = dipole_x_core - np.sum(inv_r * coords[:, 0] * weights)
    dipole_x_dft = dipole_x_core - np.sum(dft_r * coords[:, 0] * weights)
    save_data["dipole_x"] = dipole_x
    save_data["dipole_x_inv"] = dipole_x_inv
    save_data["dipole_x_dft"] = dipole_x_dft

    dipole_y_core = 0
    for i_atom in range(self.mol.natm):
        dipole_y_core += (
            self.mol.atom_charges()[i_atom] * self.mol.atom_coords()[i_atom][1]
        )
    dipole_y = dipole_y_core - np.sum(rho_cc * coords[:, 1] * weights)
    dipole_y_inv = dipole_y_core - np.sum(inv_r * coords[:, 1] * weights)
    dipole_y_dft = dipole_y_core - np.sum(dft_r * coords[:, 1] * weights)
    save_data["dipole_y"] = dipole_y
    save_data["dipole_y_inv"] = dipole_y_inv
    save_data["dipole_y_dft"] = dipole_y_dft

    dipole_z_core = 0
    for i_atom in range(self.mol.natm):
        dipole_z_core += (
            self.mol.atom_charges()[i_atom] * self.mol.atom_coords()[i_atom][2]
        )
    dipole_z = dipole_z_core - np.sum(rho_cc * coords[:, 2] * weights)
    dipole_z_inv = dipole_z_core - np.sum(inv_r * coords[:, 2] * weights)
    dipole_z_dft = dipole_z_core - np.sum(dft_r * coords[:, 2] * weights)
    save_data["dipole_z"] = dipole_z
    save_data["dipole_z_inv"] = dipole_z_inv
    save_data["dipole_z_dft"] = dipole_z_dft

    ao_value = pyscf.dft.numint.eval_ao(self.mol, coords, deriv=1)
    inv_r_3 = pyscf.dft.numint.eval_rho(self.mol, ao_value, dm1_inv, xctype="GGA")
    evxc_b3lyp = pyscf.dft.libxc.eval_xc("b3lyp", inv_r_3)
    exc_b3lyp = evxc_b3lyp[0]
    vxc_b3lyp = evxc_b3lyp[1][0]

    data_grids_norm = process_input(inv_r_3, grids)

    with open(DATA_PATH / f"save_data_{self.name}.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4)

    np.savez_compressed(
        DATA_PATH / f"data_{self.name}.npz",
        dm_cc=dm1_cc,
        dm_inv=dm1_inv,
        rho_cc=grids.vector_to_matrix(rho_cc),
        rho_inv=grids.vector_to_matrix(inv_r),
        weights=grids.vector_to_matrix(weights),
        vxc=grids.vector_to_matrix(vxc_inv),
        vxc_b3lyp=grids.vector_to_matrix(vxc_inv - vxc_b3lyp),
        vxc1_b3lyp=grids.vector_to_matrix(vxc_inv - evxc_b3lyp[0]),
        exc=grids.vector_to_matrix(exc_over_rho_grids_fake),
        exc_real=grids.vector_to_matrix(exc_over_rho_grids),
        exc_tr_b3lyp=grids.vector_to_matrix(
            exc_over_rho_grids_fake + (tau_rho_wf - tau_rho_ks) / inv_r - exc_b3lyp
        ),
        exc1_tr_b3lyp=grids.vector_to_matrix(
            exc_over_rho_grids_fake1 + (tau_rho_wf - tau_rho_ks) / inv_r - exc_b3lyp
        ),
        exc_tr=grids.vector_to_matrix(
            exc_over_rho_grids_fake + (tau_rho_wf - tau_rho_ks) / inv_r
        ),
        exc1_tr=grids.vector_to_matrix(
            exc_over_rho_grids_fake1 + (tau_rho_wf - tau_rho_ks) / inv_r
        ),
        rho_inv_4_norm=data_grids_norm,
        coords_x=grids.vector_to_matrix(coords[:, 0]),
        coords_y=grids.vector_to_matrix(coords[:, 1]),
        coords_z=grids.vector_to_matrix(coords[:, 2]),
    )
