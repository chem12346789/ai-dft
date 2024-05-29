from pathlib import Path
from itertools import product

import numpy as np
import pyscf
from pyscf import dft

from cadft.utils import gen_basis
from cadft.utils import rotate
from cadft.utils import Mol


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

        mf = pyscf.scf.RHF(self.mol)
        mf.kernel()
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot
        ej_mat_cc = np.einsum("pqrs,pq,rs->rs", eri, dm1_cc, dm1_cc)
        ek_mat_cc = np.einsum("pqrs,pr,qs->qs", eri, dm1_cc, dm1_cc)
        exc_mat = (
            np.einsum("pqrs,pqrs->rs", eri, dm2_cc)
            - np.einsum("pqrs,pq,rs->rs", eri, dm1_cc, dm1_cc)
        ) / 2

        mdft = pyscf.scf.RKS(self.mol)
        mdft.xc = xc_code
        mdft.kernel()
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        e_dft = mdft.e_tot
        coords = mdft.grids.coords
        weights = mdft.grids.weights
        ao_value = dft.numint.eval_ao(self.mol, coords, deriv=1)
        ej_mat_dft = np.einsum("pqrs,pq,rs->rs", eri, dm1_dft, dm1_dft)
        ek_mat_dft = np.einsum("pqrs,pr,qs->qs", eri, dm1_dft, dm1_dft)

        dft_mat_dft = np.zeros((self.mol.nao, self.mol.nao))
        dft_mat_cc = np.zeros((self.mol.nao, self.mol.nao))
        delta_exc_cc = np.zeros((self.mol.nao, self.mol.nao))
        cc_dft_diff = np.zeros((self.mol.nao, self.mol.nao))

        rho = dft.numint.eval_rho(self.mol, ao_value, dm1_dft, xctype="GGA")
        exc_dft_grids = dft.libxc.eval_xc("b3lyp", rho)[0]

        rho = dft.numint.eval_rho(self.mol, ao_value, dm1_cc, xctype="GGA")
        exc_cc_grids = dft.libxc.eval_xc("b3lyp", rho)[0]

        for i, j in product(range(self.mol.nao), range(self.mol.nao)):
            ao_ij = np.einsum("i,i,i->i", ao_value[0][:, i], ao_value[0][:, j], weights)
            rho = dm1_dft[i, j] * ao_ij
            dft_mat_dft[i, j] = np.einsum("i,i->", exc_dft_grids, rho)
            rho = dm1_cc[i, j] * ao_ij
            dft_mat_cc[i, j] = np.einsum("i,i->", exc_cc_grids, rho)

        dft_mat_cc -= ek_mat_cc * 0.05
        dft_mat_dft -= ek_mat_dft * 0.05
        delta_exc_cc = exc_mat - dft_mat_cc

        cc_dft_diff = (
            exc_mat
            - dft_mat_dft
            + (h1e * dm1_cc)
            - (h1e * dm1_dft)
            + ej_mat_cc * 0.5
            - ej_mat_dft * 0.5
        )

        np.savez_compressed(
            Path("data") / "dm" / (f"data_{self.name}.npz"),
            dm1_dft=dm1_dft,
            dm1_cc=dm1_cc,
            cc_dft_diff=cc_dft_diff,
            delta_exc_cc=delta_exc_cc,
            exc_mat=exc_mat,
            dft_mat_dft=dft_mat_dft,
            dft_mat_cc=dft_mat_cc,
            ej_mat_cc=ej_mat_cc,
            ej_mat_dft=ej_mat_dft,
            ek_mat_cc=ek_mat_cc,
            ek_mat_dft=ek_mat_dft,
            e_cc=e_cc,
            e_dft=e_dft,
            energy_nuc=self.mol.energy_nuc(),
            aoslice_by_atom=self.aoslice_by_atom,
        )

        print(e_dft - mdft.energy_tot(dm=dm1_dft))
        print(
            np.sum(dft_mat_dft + (h1e * dm1_dft) + ej_mat_dft * 0.5)
            - mdft.energy_elec(dm=dm1_dft)[0]
        )
        print(
            np.sum(dft_mat_cc + (h1e * dm1_cc) + ej_mat_cc * 0.5)
            - mdft.energy_elec(dm=dm1_cc)[0]
        )
        print(e_cc - e_dft - np.sum(cc_dft_diff))
