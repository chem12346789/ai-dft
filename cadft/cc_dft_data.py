import pyscf

from scipy import linalg as LA

from cadft.utils import gen_basis
from cadft.utils import rotate
from cadft.utils import mrks
from cadft.utils import save_dm1, save_dm1_dft
from cadft.utils import Mol
from cadft.utils.Grids import Grid

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

        print(molecular)
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
        Generate rho density of cc/dft and energy density of cc.
        After generating them, save them to data/grids/data_{self.name}.npz.
        """
        print(f"Save_dm1 module. Generate {self.name}")
        save_dm1(self, cc_triple, xc_code=xc_code)

    def save_dm1_dft(
        self,
        cc_triple,
        xc_code="b3lyp",
    ):
        """
        Generate rho density of cc/dft, energy density of cc and another type of energy density of dft.
        After generating them, save them to data/grids/data_{self.name}.npz.
        """
        print(f"Save_dm1_dft module. Generate {self.name}")
        save_dm1_dft(self, cc_triple, xc_code=xc_code)

    def mrks(self, frac_old, load_inv):
        """
        Generate 1-RDM.
        """
        print(f"Mrks module. Generate {self.name}")
        mrks(self, frac_old, load_inv)

    def test_mol(self):
        """
        Generate 1-RDM.
        """
        mdft = pyscf.scf.RKS(self.mol)
        mdft.xc = "b3lyp"
        mdft.kernel()

        mf = pyscf.scf.RHF(self.mol)
        mf.kernel()
        mycc = pyscf.cc.CCSD(mf)
        mycc.direct = True
        mycc.incore_complete = True
        mycc.async_io = False
        mycc.kernel()

        self.h1e = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
        self.eri = self.mol.intor("int2e")

        mat_s = self.mol.intor("int1e_ovlp")
        self.mat_hs = LA.fractional_matrix_power(mat_s, -0.5).real

        self.grids = Grid(self.mol)
        self.ao_0 = pyscf.dft.numint.eval_ao(self.mol, self.grids.coords)

        self.dm1_cc = mycc.make_rdm1(ao_repr=True)
        self.e_cc = mycc.e_tot

        self.dm1_dft = mdft.make_rdm1(ao_repr=True)
        self.e_dft = mdft.e_tot
