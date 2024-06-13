import json
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pyscf
import scipy.linalg as LA
import opt_einsum as oe

from cadft.utils.gen_tau import gen_taup_rho, gen_tau_rho
from cadft.utils.Grids import Grid

AU2KJMOL = 2625.5


def mrks(self, frac_old, load_inv=True):
    """
    Generate 1-RDM.
    """
    Path(f"data/grids/saved_data/{self.name}").mkdir(parents=True, exist_ok=True)

    mdft = pyscf.scf.RKS(self.mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    mf = pyscf.scf.RHF(self.mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()

    h1e = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
    eri = self.mol.intor("int2e")
    mo = mf.mo_coeff
    nocc = self.mol.nelec[0]
    h1_mo = np.einsum("ab,ai,bj->ij", h1e, mo, mo)
    eri_mo = pyscf.ao2mo.kernel(eri, mo, compact=False)
    norb = mo.shape[1]
    print(h1e.shape)

    mat_s = self.mol.intor("int1e_ovlp")
    mat_hs = LA.fractional_matrix_power(mat_s, -0.5).real

    dm1_cc = mycc.make_rdm1(ao_repr=True)
    dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
    dm2_cc = mycc.make_rdm2(ao_repr=True)
    dm2_cc_mo = mycc.make_rdm2(ao_repr=False)
    e_cc = mycc.e_tot

    grids = Grid(self.mol)
    coords = grids.coords
    weights = grids.weights
    ao_value = pyscf.dft.numint.eval_ao(self.mol, coords, deriv=2)

    ao_0 = ao_value[0, :, :]
    ao_1 = ao_value[1:4, :, :]
    ao_2_diag = ao_value[4, :, :] + ao_value[7, :, :] + ao_value[9, :, :]

    oe_taup_rho = oe.contract_expression(
        "pm,m,n,kpn->pk",
        ao_0,
        (norb,),
        (norb,),
        ao_1,
        constants=[0, 3],
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

    def hybrid(new, old):
        """
        Generate the hybrid density matrix.
        """
        return new * (1 - frac_old) + old * frac_old

    rho_cc = (
        pyscf.dft.numint.eval_rho(self.mol, ao_value, dm1_cc, xctype="mGGA") + 1e-14
    )
    rho_cc_half = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_cc / 2) + 1e-14
    exc_grids = np.zeros_like(rho_cc[0])

    if load_inv and Path(f"data/grids/saved_data/{self.name}/exc_grids.npy").exists():
        exc_grids = np.load(f"data/grids/saved_data/{self.name}/exc_grids.npy")
        exc_over_rho_grids = np.load(
            f"data/grids/saved_data/{self.name}/exc_over_rho_grids.npy"
        )
    else:
        print("Calculating exc_grids")
        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            0.5 * (dm2_cc - oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)),
            (self.mol.nao,),
            (self.mol.nao,),
            (self.mol.nao, self.mol.nao),
            constants=[0],
            optimize="optimal",
        )

        for i, coord in enumerate(tqdm(coords)):
            ao_0_i = ao_value[0][i]
            with self.mol.with_rinv_origin(coord):
                rinv = self.mol.intor("int1e_rinv")
                exc_grids[i] += expr_rinv_dm2_r(ao_0_i, ao_0_i, rinv, backend="torch")

        exc_over_rho_grids = exc_grids / rho_cc[0]
        ene_vc = np.sum(exc_over_rho_grids * rho_cc[0] * weights)
        error = ene_vc - (
            np.einsum("pqrs,pqrs", eri, dm2_cc).real / 2
            - np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc).real / 2
        )

        print(f"Error: {(1e3 * error):.5f} mHa")
        np.save(
            f"data/grids/saved_data/{self.name}/exc_grids.npy",
            exc_grids,
        )
        np.save(
            f"data/grids/saved_data/{self.name}/exc_over_rho_grids.npy",
            exc_over_rho_grids,
        )

    if load_inv and Path(f"data/grids/saved_data/{self.name}/emax.npy").exists():
        emax = np.load(f"data/grids/saved_data/{self.name}/emax.npy")
        taup_rho_wf = np.load(f"data/grids/saved_data/{self.name}/taup_rho_wf.npy")
        tau_rho_wf = np.load(f"data/grids/saved_data/{self.name}/tau_rho_wf.npy")
        v_vxc_e_taup = np.load(f"data/grids/saved_data/{self.name}/v_vxc_e_taup.npy")
    else:
        generalized_fock = dm1_cc_mo @ h1_mo + oe.contract(
            "rsnq,rsmq->mn", eri_mo, dm2_cc_mo
        )
        generalized_fock = 0.5 * (generalized_fock + generalized_fock.T)

        eig_e, eig_v = np.linalg.eigh(generalized_fock)
        eig_v = mo @ eig_v
        eig_e = eig_e / 2
        e_bar_r_wf = (
            oe.contract(
                "i,mi,ni,pm,pn->p",
                eig_e,
                eig_v,
                eig_v,
                ao_0,
                ao_0,
            )
            / rho_cc_half
        )

        emax = np.max(e_bar_r_wf)
        v_vxc_e_taup = -e_bar_r_wf

        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
        eigs_v_dm1 = mo @ eigs_v_dm1
        eigs_e_dm1 = eigs_e_dm1 / 2

        taup_rho_wf = gen_taup_rho(
            rho_cc_half,
            eigs_v_dm1,
            eigs_e_dm1,
            oe_taup_rho,
            backend="torch",
        )

        tau_rho_wf = gen_tau_rho(
            rho_cc_half,
            eigs_v_dm1,
            eigs_e_dm1,
            oe_tau_rho,
            backend="torch",
        )

        v_vxc_e_taup += exc_over_rho_grids * 2 + taup_rho_wf / rho_cc_half

        np.save(f"data/grids/saved_data/{self.name}/emax.npy", emax)
        np.save(f"data/grids/saved_data/{self.name}/taup_rho_wf.npy", taup_rho_wf)
        np.save(f"data/grids/saved_data/{self.name}/tau_rho_wf.npy", tau_rho_wf)
        np.save(f"data/grids/saved_data/{self.name}/v_vxc_e_taup.npy", v_vxc_e_taup)

    if load_inv and Path(f"data/grids/saved_data/{self.name}/dm1_inv.npy").exists():
        dm1_inv = np.load(f"data/grids/saved_data/{self.name}/dm1_inv.npy")
        vxc_inv = np.load(f"data/grids/saved_data/{self.name}/vxc_inv.npy")
        tau_rho_ks = np.load(f"data/grids/saved_data/{self.name}/tau_rho_ks.npy")
        taup_rho_ks = np.load(f"data/grids/saved_data/{self.name}/taup_rho_ks.npy")
    else:
        eigvecs_inv = mf.mo_energy.copy()
        mo_inv = mo.copy()
        dm1_inv = dm1_cc / 2
        vj_inv = mf.get_jk(self.mol, dm1_inv * 2, 1)[0]
        vxc_inv = pyscf.dft.libxc.eval_xc(
            "B88,P86",
            pyscf.dft.numint.eval_rho(
                self.mol,
                ao_value[:4, :, :],
                dm1_inv * 2,
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

        for i in range(25000):
            vj_inv = hybrid(mf.get_jk(self.mol, 2 * dm1_inv, 1)[0], vj_inv)
            dm1_inv_r = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv) + 1e-14

            potential_shift = emax - np.max(eigvecs_inv[:nocc])
            ebar_ks = (
                oe_ebar_r_ks(
                    eigvecs_inv[:nocc] + potential_shift,
                    mo_inv[:, :nocc],
                    mo_inv[:, :nocc],
                )
                / dm1_inv_r
            )

            taup_rho_ks = gen_taup_rho(
                dm1_inv_r,
                mo_inv[:, :nocc],
                np.ones(nocc),
                oe_taup_rho,
                backend="torch",
            )

            vxc_inv_old = vxc_inv.copy()
            vxc_inv = v_vxc_e_taup + ebar_ks - taup_rho_ks / dm1_inv_r
            error_vxc = np.linalg.norm((vxc_inv - vxc_inv_old) * weights)
            vxc_inv = hybrid(vxc_inv, vxc_inv_old)

            xc_v = oe_fock(vxc_inv, weights, backend="torch")
            eigvecs_inv, mo_inv = np.linalg.eigh(
                mat_hs @ (h1e + vj_inv + xc_v) @ mat_hs
            )
            mo_inv = mat_hs @ mo_inv
            dm1_inv_old = dm1_inv.copy()
            dm1_inv = mo_inv[:, :nocc] @ mo_inv[:, :nocc].T
            error_dm1 = np.linalg.norm(dm1_inv - dm1_inv_old)

            if i % 100 == 0:
                print(
                    f"step:{i:<8}",
                    f"error of vxc: {error_vxc::<10.5e}",
                    f"dm: {error_dm1::<10.5e}",
                    f"shift: {potential_shift::<10.5e}",
                )
            if (i > 0) and (error_vxc < 1e-6):
                print(
                    f"step:{i:<8}",
                    f"error of vxc: {error_vxc::<10.5e}",
                    f"dm: {error_dm1::<10.5e}",
                    f"shift: {potential_shift::<10.5e}",
                )
                break

        tau_rho_ks = gen_tau_rho(
            dm1_inv_r,
            mo_inv[:, :nocc],
            np.ones(nocc),
            oe_tau_rho,
            backend="torch",
        )
        np.save(f"data/grids/saved_data/{self.name}/dm1_inv.npy", dm1_inv)
        np.save(f"data/grids/saved_data/{self.name}/vxc_inv.npy", vxc_inv)
        np.save(f"data/grids/saved_data/{self.name}/tau_rho_ks.npy", tau_rho_ks)
        np.save(f"data/grids/saved_data/{self.name}/taup_rho_ks.npy", taup_rho_ks)

    kin_correct = 2 * np.sum((tau_rho_wf - tau_rho_ks) * weights)
    kin_correct1 = 2 * np.sum((taup_rho_wf - taup_rho_ks) * weights)
    inv_r = pyscf.dft.numint.eval_rho(self.mol, ao_0, dm1_inv * 2) + 1e-14
    dft_r = pyscf.dft.numint.eval_rho(self.mol, ao_0, mdft.make_rdm1()) + 1e-14
    exc_over_rho_grids_real = exc_over_rho_grids.copy()

    if True:
        expr_rinv_dm1_r = oe.contract_expression(
            "ij,ij->",
            dm1_cc,
            (norb, norb),
            constants=[0],
            optimize="optimal",
        )

        for i, coord in enumerate(tqdm(coords)):
            with self.mol.with_rinv_origin(coord):
                rinv = self.mol.intor("int1e_rinv")
                rinv_dm1_r = expr_rinv_dm1_r(rinv, backend="torch")
                exc_grids[i] += rinv_dm1_r * rho_cc[0][i] - rinv_dm1_r * inv_r[i]

            for i_atom in range(self.mol.natm):
                distance = np.linalg.norm(self.mol.atom_coords()[i_atom] - coord)
                cut_off = 1e-3
                if distance < cut_off:
                    # distance = 2 / cut_off - 1 / cut_off / cut_off * distance
                    distance = cut_off
                exc_grids[i] -= (
                    (rho_cc[0][i] - inv_r[i])
                    * self.mol.atom_charges()[i_atom]
                    / distance
                )
        exc_over_rho_grids = exc_grids / inv_r

    save_data = {}
    save_data["energy"] = AU2KJMOL * e_cc
    save_data["correct kinetic energy"] = AU2KJMOL * kin_correct
    save_data["correct kinetic energy1"] = AU2KJMOL * kin_correct1
    save_data["energy_dft"] = AU2KJMOL * (mdft.e_tot - e_cc)
    save_data["energy_inv"] = AU2KJMOL * (
        (
            oe.contract("ij,ji->", h1e, dm1_inv * 2)
            + 0.5 * oe.contract("pqrs,pq,rs->", eri, dm1_inv * 2, dm1_inv * 2)
            + self.mol.energy_nuc()
            + np.sum(exc_over_rho_grids * inv_r * weights)
            + kin_correct
        )
        - e_cc
    )
    save_data["energy_inv1"] = AU2KJMOL * (
        (
            oe.contract("ij,ji->", h1e, dm1_inv * 2)
            + 0.5 * oe.contract("pqrs,pq,rs->", eri, dm1_inv * 2, dm1_inv * 2)
            + self.mol.energy_nuc()
            + np.sum(exc_over_rho_grids * inv_r * weights)
            + kin_correct1
        )
        - e_cc
    )

    error_inv_r = np.sum(np.abs(inv_r - rho_cc[0]) * weights)
    error_dft_r = np.sum(np.abs(dft_r - rho_cc[0]) * weights)
    save_data["error of dm1_inv"] = error_inv_r
    save_data["error of dm1_dft"] = error_dft_r

    dipole_x = np.sum(rho_cc[0] * coords[:, 0] * weights)
    dipole_x_inv = np.sum(inv_r * coords[:, 0] * weights)
    dipole_x_dft = np.sum(dft_r * coords[:, 0] * weights)
    dipole_y = np.sum(rho_cc[0] * coords[:, 1] * weights)
    dipole_y_inv = np.sum(inv_r * coords[:, 1] * weights)
    dipole_y_dft = np.sum(dft_r * coords[:, 1] * weights)
    dipole_z = np.sum(rho_cc[0] * coords[:, 2] * weights)
    dipole_z_inv = np.sum(inv_r * coords[:, 2] * weights)
    dipole_z_dft = np.sum(dft_r * coords[:, 2] * weights)
    save_data["dipole_x"] = dipole_x
    save_data["dipole_x_inv"] = dipole_x_inv
    save_data["dipole_x_dft"] = dipole_x_dft
    save_data["dipole_y"] = dipole_y
    save_data["dipole_y_inv"] = dipole_y_inv
    save_data["dipole_y_dft"] = dipole_y_dft
    save_data["dipole_z"] = dipole_z
    save_data["dipole_z_inv"] = dipole_z_inv
    save_data["dipole_z_dft"] = dipole_z_dft

    with open(
        Path("data") / "grids" / f"save_data_{self.name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(save_data, f, indent=4)

    np.savez_compressed(
        Path("data") / "grids" / f"data_{self.name}.npz",
        rho_cc=grids.vector_to_matrix(rho_cc[0]),
        weights=grids.vector_to_matrix(weights),
        exc=grids.vector_to_matrix(exc_over_rho_grids),
        vxc=grids.vector_to_matrix(vxc_inv),
        exc_real=grids.vector_to_matrix(exc_over_rho_grids_real),
        coords=coords,
        coords_x=grids.vector_to_matrix(coords[:, 0]),
        coords_y=grids.vector_to_matrix(coords[:, 1]),
        coords_z=grids.vector_to_matrix(coords[:, 2]),
    )
