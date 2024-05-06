from pathlib import Path
import json

import numpy as np

import pyscf

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
        )

        aoslice_by_atom = self.mol.aoslice_by_atom()[:, 2:]
        self.atom_info = {"slice": {}, "atom": {}, "nao": {}}
        for i in range(self.mol.natm):
            self.atom_info["slice"][i] = slice(
                aoslice_by_atom[i][0], aoslice_by_atom[i][1]
            )
            self.atom_info["atom"][i] = molecular[i][0]
            self.atom_info["nao"][i] = aoslice_by_atom[i][1] - aoslice_by_atom[i][0]

    def save_dm1(self, cc_triple=False, xc_code="b3lyp"):
        """
        Generate 1-RDM.
        """
        mf = pyscf.scf.RHF(self.mol)
        mf.kernel()
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()

        mdft = pyscf.scf.RKS(self.mol)
        mdft.xc = xc_code
        mdft.kernel()

        dm1_cc = mycc.make_rdm1(ao_repr=True)
        e_cc = mycc.e_tot
        dm1_dft = mdft.make_rdm1(ao_repr=True)

        dm2_cc = mycc.make_rdm2(ao_repr=True)
        eri = self.mol.intor("int2e")
        exc_mat = (
            np.einsum("pqrs,pqrs->rs", eri, dm2_cc)
            - np.einsum("pqrs,pq,rs->rs", eri, dm1_cc, dm1_cc)
        ) / 2
        h1e = self.mol.intor("int1e_nuc") + self.mol.intor("int1e_kin")
        print(
            e_cc,
            np.sum(exc_mat)
            + np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc) / 2
            + np.sum(h1e * dm1_cc)
            + self.mol.energy_nuc(),
        )
        data_path = Path("data")
        weight_path = data_path / "weight"
        weight_path.mkdir(parents=True, exist_ok=True)

        np.save(weight_path / f"e_ccsd_{self.name}.npy", e_cc)
        np.save(weight_path / f"energy_nuc_{self.name}.npy", self.mol.energy_nuc())

        save_json = {}
        save_json["atom"] = self.mol.atom
        save_json["basis"] = self.mol.basis
        with open(
            weight_path / f"mol_info_{self.name}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(save_json, f, indent=4)

        for i in range(self.mol.natm):
            for j in range(self.mol.natm):
                atom_name = self.atom_info["atom"][i] + self.atom_info["atom"][j]
                input_path = data_path / atom_name / "input"
                output_path = data_path / atom_name / "output"
                input_path.mkdir(parents=True, exist_ok=True)
                output_path.mkdir(parents=True, exist_ok=True)

                input_mat = dm1_dft[
                    self.atom_info["slice"][i], self.atom_info["slice"][j]
                ]
                output_dm1_mat = dm1_cc[
                    self.atom_info["slice"][i], self.atom_info["slice"][j]
                ]

                np.save(
                    input_path / f"input_{self.name}_{i}_{j}.npy",
                    input_mat,
                )
                np.save(
                    output_path / f"output_dm1_{self.name}_{i}_{j}.npy",
                    output_dm1_mat,
                )

                output_exc_mat = exc_mat[
                    self.atom_info["slice"][i], self.atom_info["slice"][j]
                ]
                np.save(
                    output_path / f"output_exc_{self.name}_{i}_{j}.npy",
                    output_exc_mat,
                )
