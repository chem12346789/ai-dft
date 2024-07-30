from pathlib import Path
from timeit import default_timer as timer

import pyscf
import numpy as np
from scipy import linalg as LA

from cadft.utils import gen_basis
from cadft.utils import rotate
from cadft.utils import mrks, mrks_append
from cadft.utils import Mol
from cadft.utils.Grids import Grid
from cadft.utils import MAIN_PATH, DATA_CC_PATH

AU2KJMOL = 2625.5


class CC_DFT_DATA:

    def __init__(
        self,
        molecular=Mol["methane"],
        name="methane",
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
        print(self.mol.atom)

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

    def mrks(self, frac_old, load_inv):
        """
        Generate 1-RDM.
        """
        print(f"Mrks module. Generate {self.name}")
        mrks(self, frac_old, load_inv)

    def mrks_append(self, frac_old, load_inv):
        """
        Generate 1-RDM.
        """
        print(f"Mrks_append module. Generate {self.name}")
        mrks_append(self, frac_old, load_inv)

    # pylint: disable=W0201
    def test_mol(self):
        """
        Generate 1-RDM.
        """
        # if False:
        if (DATA_CC_PATH / f"data_{self.name}.npz").exists():
            print(f"Load data from {DATA_CC_PATH}/data_{self.name}.npz")
            data_saved = np.load(f"{DATA_CC_PATH}/data_{self.name}.npz")
            self.dm1_cc = data_saved["dm1_cc"]
            self.e_cc = data_saved["e_cc"]
            self.time_cc = data_saved["time_cc"]

            if "dm1_dft" in data_saved.files:
                self.dm1_dft = data_saved["dm1_dft"]
                self.e_dft = data_saved["e_dft"]
                self.time_dft = data_saved["time_dft"]
                self.h1e = data_saved["h1e"]
                self.mat_s = data_saved["mat_s"]
                self.mat_hs = data_saved["mat_hs"]
                self.dm1_hf = data_saved["dm1_hf"]
            else:
                time_start = timer()
                mdft = pyscf.scf.RKS(self.mol)
                mdft.xc = "b3lyp"
                mdft.kernel()
                self.dm1_dft = mdft.make_rdm1(ao_repr=True)
                self.e_dft = mdft.e_tot
                self.time_dft = timer() - time_start

                mf = pyscf.scf.RHF(self.mol)
                mf.kernel()
                self.dm1_hf = mf.make_rdm1(ao_repr=True)

                self.h1e = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
                self.mat_s = self.mol.intor("int1e_ovlp")
                self.mat_hs = LA.fractional_matrix_power(self.mat_s, -0.5).real

                np.savez_compressed(
                    Path(f"{MAIN_PATH}/data/test/data_{self.name}.npz"),
                    dm1_cc=self.dm1_cc,
                    e_cc=self.e_cc,
                    time_cc=self.time_cc,
                    dm1_dft=self.dm1_dft,
                    e_dft=self.e_dft,
                    time_dft=self.time_dft,
                    h1e=self.h1e,
                    mat_s=self.mat_s,
                    mat_hs=self.mat_hs,
                    dm1_hf=self.dm1_hf,
                )
        else:
            print(f"Generate data for {self.name}")
            time_start = timer()
            mf = pyscf.scf.RHF(self.mol)
            mf.kernel()
            mycc = pyscf.cc.CCSD(mf)
            mycc.direct = True
            mycc.incore_complete = True
            mycc.async_io = False
            mycc.kernel()
            self.dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
            self.time_cc = timer() - time_start

            self.dm1_hf = mf.make_rdm1(ao_repr=True)

            time_start = timer()
            mdft = pyscf.scf.RKS(self.mol)
            mdft.xc = "b3lyp"
            mdft.kernel()
            self.dm1_dft = mdft.make_rdm1(ao_repr=True)
            self.e_dft = mdft.e_tot
            self.time_dft = timer() - time_start

            self.h1e = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
            self.mat_s = self.mol.intor("int1e_ovlp")
            self.mat_hs = LA.fractional_matrix_power(self.mat_s, -0.5).real

            np.savez_compressed(
                Path(f"{MAIN_PATH}/data/test/data_{self.name}.npz"),
                dm1_cc=self.dm1_cc,
                e_cc=self.e_cc,
                time_cc=self.time_cc,
                dm1_dft=self.dm1_dft,
                e_dft=self.e_dft,
                time_dft=self.time_dft,
                h1e=self.h1e,
                mat_s=self.mat_s,
                mat_hs=self.mat_hs,
                dm1_hf=self.dm1_hf,
            )

        self.grids = Grid(self.mol)
        self.ao_0 = pyscf.dft.numint.eval_ao(self.mol, self.grids.coords)
        self.ao_1 = pyscf.dft.numint.eval_ao(self.mol, self.grids.coords, deriv=1)
