from pathlib import Path
from itertools import product
from tqdm import tqdm

import numpy as np
import pyscf
from pyscf import dft
import opt_einsum as oe
import torch

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
        cc_triple=False,
        xc_code="b3lyp",
    ):
        """
        Generate 1-RDM.
        """
        h1e = self.mol.intor("int1e_nuc") + self.mol.intor("int1e_kin")
        eri = self.mol.intor("int2e")

        mdft = pyscf.scf.RKS(self.mol)
        mdft.xc = xc_code
        mdft.kernel()
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        print(np.shape(dm1_dft))

        mf = pyscf.scf.RHF(self.mol)
        mf.kernel()
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot

        grids = Grid(self.mol)
        coords = grids.coords
        weights = grids.weights
        ao_value = dft.numint.eval_ao(self.mol, coords, deriv=1)

        mf = pyscf.scf.RHF(self.mol)
        mf.kernel()
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot

        rho_cc = dft.numint.eval_rho(self.mol, ao_value, dm1_cc, xctype="GGA")
        rho_dft = dft.numint.eval_rho(self.mol, ao_value, dm1_dft, xctype="GGA")

        exc_over_dm_cc_grids = -dft.libxc.eval_xc("b3lyp", rho_cc)[0]
        cc_dft_ene = (
            np.sum(-exc_over_dm_cc_grids * rho_cc[0] * weights)
            - np.einsum("pqrs,pr,qs->", eri, dm1_cc, dm1_cc) * 0.05
        )

        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            process(
                0.5 * (dm2_cc - oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc))
                + 0.05 * oe.contract("pr,qs->pqrs", dm1_cc, dm1_cc)
            ),
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
                    expr_rinv_dm2_r(
                        process(ao_0_i),
                        process(ao_0_i),
                        process(rinv),
                        backend="torch",
                    )
                    / rho_cc[0][i]
                )

        ene_vc = np.sum(exc_over_dm_cc_grids * rho_cc[0] * weights)
        error = (
            ene_vc
            + cc_dft_ene
            + np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc) / 2
            + np.sum(h1e * dm1_cc)
            + self.mol.energy_nuc()
            - e_cc
        )
        print(f"Error: {(1e3 * error):.5f} mHa")

        np.savez_compressed(
            Path("data") / "grids" / (f"data_{self.name}.npz"),
            rho_dft=grids.vector_to_matrix(rho_dft[0]),
            rho_cc=grids.vector_to_matrix(rho_cc[0]),
            exc_over_dm_cc_grids=grids.vector_to_matrix(exc_over_dm_cc_grids),
            weights=grids.vector_to_matrix(weights),
        )

        # rho_dft = dft.numint.eval_rho(self.mol, ao_value, dm1_dft, xctype="GGA")
        # rho_cc = dft.numint.eval_rho(self.mol, ao_value, dm1_cc, xctype="GGA")
        # delta_exc_over_dm_dft_grids = -dft.libxc.eval_xc("b3lyp", rho_dft)[0]

        # expr_rinv_dm2_r = oe.contract_expression(
        #     "ijkl,i,j,kl->",
        #     0.5 * dm2_cc
        #     - (
        #         0.5 * oe.contract("pq,rs->pqrs", dm1_dft, dm1_dft)
        #         - 0.05 * oe.contract("pr,qs->pqrs", dm1_dft, dm1_dft)
        #     ),
        #     (self.mol.nao,),
        #     (self.mol.nao,),
        #     (self.mol.nao, self.mol.nao),
        #     constants=[0],
        #     optimize="optimal",
        # )

        # for i, coord in enumerate(tqdm(coords)):
        #     ao_0_i = ao_value[0][i]
        #     if np.linalg.norm(ao_0_i) < 1e-10:
        #         continue
        #     if np.linalg.norm(rho_dft[0][i]) < 1e-10:
        #         continue
        #     with self.mol.with_rinv_origin(coord):
        #         rinv = self.mol.intor("int1e_rinv")
        #         delta_exc_over_dm_dft_grids[i] += (
        #             expr_rinv_dm2_r(ao_0_i, ao_0_i, rinv) / rho_dft[0][i]
        #         )

        #     for i_atom in range(self.mol.natm):
        #         delta_exc_over_dm_dft_grids[i] -= (
        #             (rho_cc[0][i] - rho_dft[0][i])
        #             * self.mol.atom_charges()[i_atom]
        #             / rho_dft[0][i]
        #             / np.linalg.norm(self.mol.atom_coords()[i_atom] - coord)
        #         )

        # eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc)
        # eigs_v_dm1 = self.mol.mo @ eigs_v_dm1
        # eigs_e_dm1 = eigs_e_dm1
        # tau_rho_wf = gen_tau_rho(
        #     self.aux_function,
        #     dm1_r,
        #     eigs_v_dm1,
        #     eigs_e_dm1,
        #     backend="torch",
        #     logger=self.logger,
        # )
        # delta_exc_over_dm_dft_grids = tau_rho_wf - oe_tau_rho(dm1_dft)

        # error = (
        #     np.sum(delta_exc_over_dm_dft_grids * rho_dft[0] * weights)
        #     + np.sum(self.mol.intor("int1e_kin") * dm1_cc)
        #     - np.sum(self.mol.intor("int1e_kin") * dm1_dft)
        #     - (e_cc - e_dft)
        # )
        # print(f"Error: {(1e3 * error):.5f} mHa")
        # print(
        #     f"max of delta_exc_over_dm_dft_grids: {np.max(delta_exc_over_dm_dft_grids)}"
        # )
