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

        exc_mat = (
            np.einsum("pqrs,pqrs->rs", eri, dm2_cc)
            - np.einsum("pqrs,pq,rs->rs", eri, dm1_cc, dm1_cc)
        ) / 2

        exc_mat_dft = np.zeros((self.mol.nao, self.mol.nao))
        delta_exc_cc = np.zeros((self.mol.nao, self.mol.nao))
        cc_dft_diff = np.zeros((self.mol.nao, self.mol.nao))

        rho = dft.numint.eval_rho(self.mol, ao_value, dm1_dft, xctype="GGA")
        exc_dft_grids = dft.libxc.eval_xc("b3lyp", rho)[0]
        exc_dft = (
            np.einsum("i,i,i->", exc_dft_grids, rho[0], weights)
            - np.sum(ek_mat_dft) * 0.05
        )

        rho = dft.numint.eval_rho(self.mol, ao_value, dm1_cc, xctype="GGA")
        exc_cc_grids = dft.libxc.eval_xc("b3lyp", rho)[0]
        exc_cc = (
            np.einsum("i,i,i->", exc_cc_grids, rho[0], weights)
            - np.sum(ek_mat_cc) * 0.05
        )

        for i_atom, j_atom in product(range(self.mol.natm), range(self.mol.natm)):
            slice_ = (self.atom_info["slice"][i_atom], self.atom_info["slice"][j_atom])
            dft_mat = dm1_dft[slice_]
            cc_mat = dm1_cc[slice_]

            for i, j in product(
                range(self.atom_info["nao"][i_atom]),
                range(self.atom_info["nao"][j_atom]),
            ):
                new_dm_dft = np.zeros_like(dm1_dft)
                new_dm_dft[slice_][i, j] = dft_mat[i, j]
                rho = dft.numint.eval_rho(self.mol, ao_value[0], new_dm_dft)
                exc_mat_dft[slice_][i, j] = np.einsum(
                    "i,i,i->", exc_dft_grids, rho, weights
                )

                new_dm_cc = np.zeros_like(dm1_cc)
                new_dm_cc[slice_][i, j] = cc_mat[i, j]
                rho = dft.numint.eval_rho(self.mol, ao_value[0], new_dm_cc)
                delta_exc_cc[slice_][i, j] = -np.einsum(
                    "i,i,i->", exc_cc_grids, rho, weights
                )

            delta_exc_cc[slice_] += exc_mat[slice_]
            delta_exc_cc[slice_] += ek_mat_cc[slice_] * 0.05
            exc_mat_dft[slice_] -= ek_mat_dft[slice_] * 0.05

            cc_dft_diff[slice_] = (
                exc_mat[slice_]
                - exc_mat_dft[slice_]
                + (h1e * dm1_cc)[slice_]
                - (h1e * dm1_dft)[slice_]
                + ej_mat_cc[slice_] * 0.5
                - ej_mat_dft[slice_] * 0.5
            )

        np.save(
            Path("data") / "input" / f"input_dft_{self.name}.npy",
            dm1_dft,
        )
        np.save(
            Path("data") / "input" / f"input_cc_{self.name}.npy",
            dm1_cc,
        )

        np.save(
            Path("data") / "output" / f"output_cc_dft_diff_{self.name}.npy",
            cc_dft_diff,
        )
        np.save(
            Path("data") / "output" / f"output_delta_exc_cc_{self.name}.npy",
            delta_exc_cc,
        )

        np.save(Path("data") / "weight" / f"e_ccsd_{self.name}.npy", e_cc)
        np.save(Path("data") / "weight" / f"e_dft_{self.name}.npy", e_dft)
        np.save(
            Path("data") / "weight" / f"energy_nuc_{self.name}.npy",
            self.mol.energy_nuc(),
        )
        np.save(
            Path("data") / "weight" / f"aoslice_by_atom_{self.name}.npy",
            self.aoslice_by_atom,
        )

        print(
            exc_cc
            + np.sum(delta_exc_cc)
            + np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc) / 2
            + np.sum(h1e * dm1_cc)
            + self.mol.energy_nuc()
            - e_cc
        )
        print(exc_dft - np.sum(exc_mat_dft))
        print(e_cc - e_dft - np.sum(cc_dft_diff))
