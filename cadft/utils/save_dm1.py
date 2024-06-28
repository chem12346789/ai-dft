from tqdm import tqdm
from pathlib import Path

import numpy as np
import pyscf
import opt_einsum as oe

from cadft.utils.Grids import Grid
from cadft.utils.env_var import MAIN_PATH


def save_dm1(
    self,
    cc_triple,
    xc_code="b3lyp",
):
    """
    Generate 1-RDM.
    """
    mdft = pyscf.scf.RKS(self.mol)
    mdft.xc = xc_code
    mdft.kernel()
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    e_dft = mdft.e_tot
    print(np.shape(dm1_dft))

    mf = pyscf.scf.RHF(self.mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()
    dm1_cc = mycc.make_rdm1(ao_repr=True)
    e_cc = mycc.e_tot
    e_cc_dft = mdft.energy_tot(dm1_cc)

    grids = Grid(self.mol)
    coords = grids.coords
    weights = grids.weights
    ao_2 = pyscf.dft.numint.eval_ao(self.mol, coords, deriv=2)
    ao_0 = ao_2[0, :, :]
    ao_value = ao_2[:4, :, :]
    ao_2_diag = ao_2[4, :, :] + ao_2[7, :, :] + ao_2[9, :, :]

    rho_dft = pyscf.dft.numint.eval_rho(self.mol, ao_value, dm1_dft, xctype="GGA")
    rho_cc = pyscf.dft.numint.eval_rho(self.mol, ao_value, dm1_cc, xctype="GGA")
    exc_over_dm_cc_grids = np.zeros_like(rho_dft[0])

    dm2_cc = mycc.make_rdm2(ao_repr=True)
    exc_over_dm_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0]
    expr_rinv_dm2_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.5 * (dm2_cc - oe.contract("pq,rs->pqrs", dm1_dft, dm1_dft))
        + 0.05 * oe.contract("pr,qs->pqrs", dm1_dft, dm1_dft),
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
            exc_over_dm_cc_grids[i] += expr_rinv_dm2_r(
                ao_0_i, ao_0_i, rinv, backend="torch"
            )

        for i_atom in range(self.mol.natm):
            distance = np.linalg.norm(self.mol.atom_coords()[i_atom] - coord)
            cut_off = 5e-3
            if distance < cut_off:
                # distance = 2 / cut_off - 1 / cut_off / cut_off * distance
                distance = cut_off
            exc_over_dm_cc_grids[i] -= (
                (rho_cc[0][i] - rho_dft[0][i])
                * self.mol.atom_charges()[i_atom]
                / distance
            )

    dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
    eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
    eigs_v_dm1 = mf.mo_coeff @ eigs_v_dm1
    for i in range(np.shape(eigs_v_dm1)[1]):
        part = oe.contract(
            "pm,m,n,pn->p",
            ao_0,
            eigs_v_dm1[:, i],
            eigs_v_dm1[:, i],
            ao_2_diag,
        )
        exc_over_dm_cc_grids -= part * eigs_e_dm1[i] / 2

    for i in range(self.mol.nelec[0]):
        part = oe.contract(
            "pm,m,n,pn->p",
            ao_0,
            mdft.mo_coeff[:, i],
            mdft.mo_coeff[:, i],
            ao_2_diag,
        )
        exc_over_dm_cc_grids += part

    error = np.sum(exc_over_dm_cc_grids * weights) - (e_cc - e_dft)

    print(f"Error: {(1e3 * error):.5f} mHa")

    np.savez_compressed(
        Path(f"{MAIN_PATH}/data") / "grids" / (f"data_{self.name}.npz"),
        rho_dft=grids.vector_to_matrix(rho_dft[0]),
        rho_cc=grids.vector_to_matrix(rho_cc[0]),
        weights=grids.vector_to_matrix(weights),
        delta_ene_cc=e_cc - e_cc_dft,
        delta_ene_dft=e_cc - e_dft,
        ene_cc=e_cc,
        exc_over_dm_cc_grids=grids.vector_to_matrix(exc_over_dm_cc_grids),
    )


def save_dm1_dft(
    self,
    cc_triple,
    xc_code="b3lyp",
):
    """
    Generate 1-RDM.
    """
    mdft = pyscf.scf.RKS(self.mol)
    mdft.xc = xc_code
    mdft.kernel()

    mf = pyscf.scf.RHF(self.mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()
    dm1_cc = mycc.make_rdm1(ao_repr=True)
    e_cc = mycc.e_tot
    e_cc_dft = mdft.energy_tot(dm1_cc)

    grids = Grid(self.mol)
    coords = grids.coords
    weights = grids.weights
    ao_value = pyscf.dft.numint.eval_ao(self.mol, coords, deriv=2)

    rho_cc = pyscf.dft.numint.eval_rho(self.mol, ao_value, dm1_cc, xctype="mGGA")

    slater_cc_grids = pyscf.dft.libxc.eval_xc("SLATER", rho_cc)[0]
    b88_cc_grids = pyscf.dft.libxc.eval_xc("B88", rho_cc)[0]
    pbex_cc_grids = pyscf.dft.libxc.eval_xc("PBE", rho_cc)[0]

    scan_cc_grids = pyscf.dft.libxc.eval_xc(",SCAN", rho_cc)[0]
    pbec_cc_grids = pyscf.dft.libxc.eval_xc(",PBE", rho_cc)[0]
    p86_cc_grids = pyscf.dft.libxc.eval_xc(",P86", rho_cc)[0]
    lyp_cc_grids = pyscf.dft.libxc.eval_xc(",LYP", rho_cc)[0]
    vwn_cc_grids = pyscf.dft.libxc.eval_xc(",VWNRPA", rho_cc)[0]

    # b3lyp_cc_grids = pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0]

    dm2_cc = mycc.make_rdm2(ao_repr=True)
    exc_over_dm_cc_grids = np.zeros_like(rho_cc[0])
    hf_over_dm_cc_grids = np.zeros_like(rho_cc[0])

    expr_rinv_dm2_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.5 * (dm2_cc - oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)),
        (self.mol.nao,),
        (self.mol.nao,),
        (self.mol.nao, self.mol.nao),
        constants=[0],
        optimize="optimal",
    )

    expr_rinv_dm1_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.25 * oe.contract("pr,qs->pqrs", dm1_cc, dm1_cc),
        (self.mol.nao,),
        (self.mol.nao,),
        (self.mol.nao, self.mol.nao),
        constants=[0],
        optimize="optimal",
    )

    for i, coord in enumerate(tqdm(coords)):
        ao_0_i = ao_value[0][i]
        if np.linalg.norm(ao_0_i) < 1e-10:
            continue
        with self.mol.with_rinv_origin(coord):
            rinv = self.mol.intor("int1e_rinv")
            exc_over_dm_cc_grids[i] += (
                expr_rinv_dm2_r(ao_0_i, ao_0_i, rinv, backend="torch") / rho_cc[0][i]
            )
            hf_over_dm_cc_grids[i] += (
                expr_rinv_dm1_r(ao_0_i, ao_0_i, rinv, backend="torch") / rho_cc[0][i]
            )

    h1e = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
    eri = self.mol.intor("int2e")
    exa_ene = (
        self.mol.energy_nuc()
        + np.einsum("ij,ij->", h1e, dm1_cc)
        + 0.5 * np.einsum("ijkl,ij,kl->", eri, dm1_cc, dm1_cc)
    )
    error_dft = (
        np.sum(
            (
                -0.2 * hf_over_dm_cc_grids
                # + b3lyp_cc_grids
                + 0.08 * slater_cc_grids
                + 0.72 * b88_cc_grids
                + 0.81 * lyp_cc_grids
                + 0.19 * vwn_cc_grids
            )
            * rho_cc[0]
            * weights
        )
        + exa_ene
        - e_cc_dft
    )
    error_cc = np.sum(exc_over_dm_cc_grids * rho_cc[0] * weights) + exa_ene - e_cc
    print(
        f"Error DFT: {(1e3 * error_dft):.5f} mHa, Error CC: {(1e3 * error_cc):.5f} mHa"
    )

    data = np.load(Path(f"{MAIN_PATH}/data") / "grids" / (f"data_{self.name}.npz"))

    np.savez_compressed(
        Path(f"{MAIN_PATH}/data") / "grids" / (f"data_{self.name}.npz"),
        rho_cc=data["rho_cc"],
        weights=data["weights"],
        exc_over_dm_cc_grids=data["exc_over_dm_cc_grids"],
        hf_over_dm_cc_grids=data["hf_over_dm_cc_grids"],
        slater_cc_grids=data["slater_cc_grids"],
        b88_cc_grids=data["b88_cc_grids"],
        lyp_cc_grids=data["lyp_cc_grids"],
        vwn_cc_grids=data["vwn_cc_grids"],
        pbex_cc_grids=grids.vector_to_matrix(pbex_cc_grids),
        pbec_cc_grids=grids.vector_to_matrix(pbec_cc_grids),
        scan_cc_grids=grids.vector_to_matrix(scan_cc_grids),
        p86_cc_grids=grids.vector_to_matrix(p86_cc_grids),
        e_cc=data["e_cc"],
    )
