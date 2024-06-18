import pyscf

from cadft.utils import gen_basis
from cadft.utils import rotate
from cadft.utils import mrks
from cadft.utils import save_dm1, save_dm1_dft
from cadft.utils import Mol

AU2KJMOL = 2625.5


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

        # rotate(molecular)

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
        save_dm1_dft(self, cc_triple, xc_code=xc_code)

    def mrks(self, frac_old, load_inv):
        """
        Generate 1-RDM.
        """
        mrks(self, frac_old, load_inv)
