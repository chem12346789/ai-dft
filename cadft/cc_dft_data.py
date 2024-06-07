from pathlib import Path
from tqdm import tqdm

import numpy as np
import pyscf
from pyscf import dft
import torch
import opt_einsum as oe

from cadft.utils import gen_basis
from cadft.utils import rotate
from cadft.utils import Mol
from cadft.utils import Grid


def process(data, device="cuda"):
    """
    Load the whole data to the device.
    """
    return torch.as_tensor(data).to(torch.float64).contiguous().to(device=device)


class CC_DFT_DATA:
    def __init__(
        self,
        molecular=Mol["Methane"],
        name="Methane",
        basis="sto-3g",
        if_basis_str=False,
    ):
        self.name = name
        self.basis = basis
        self.if_basis_str = if_basis_str

        rotate(molecular)

        self.mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(molecular, self.basis, self.if_basis_str),
            verbose=0,
        )

        self.aoslice_by_atom = self.mol.aoslice_by_atom()[:, 2:]
        self.atom_info = {"slice": {}, "atom": {}, "nao": {}}
        for i in range(self.mol.natm):
            self.atom_info["slice"][i] = slice(
                self.aoslice_by_atom[i][0], self.aoslice_by_atom[i][1]
            )
            self.atom_info["atom"][i] = molecular[i][0]
            self.atom_info["nao"][i] = (
                self.aoslice_by_atom[i][1] - self.aoslice_by_atom[i][0]
            )

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
        exc_over_dm_cc_grids = (
            -pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0]
        )
        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            0.5 * (dm2_cc - oe.contract("pq,rs->pqrs", dm1_dft, dm1_dft)),
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
                exc_over_dm_cc_grids[i] += expr_rinv_dm2_r(
                    ao_0_i, ao_0_i, rinv, backend="torch"
                )

            for i_atom in range(self.mol.natm):
                exc_over_dm_cc_grids[i] -= (
                    (rho_cc[0][i] - rho_dft[0][i])
                    * self.mol.atom_charges()[i_atom]
                    / np.linalg.norm(self.mol.atom_coords()[i_atom] - coord)
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
            Path("data") / "grids" / (f"data_{self.name}.npz"),
            rho_dft=grids.vector_to_matrix(rho_dft[0]),
            rho_cc=grids.vector_to_matrix(rho_cc[0]),
            weights=grids.vector_to_matrix(weights),
            delta_ene_cc=e_cc - e_cc_dft,
            delta_ene_dft=e_cc - e_dft,
            exc_over_dm_cc_grids=grids.vector_to_matrix(exc_over_dm_cc_grids),
        )
