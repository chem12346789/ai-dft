from timeit import default_timer as timer

import pyscf
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad import uccsd as uccsd_grad

from cadft.utils import gen_basis
from cadft.utils import mrks_diis, umrks_diis, gmrks_diis
from cadft.utils import Grid
from cadft.utils import Mol


class CC_DFT_DATA:

    def __init__(
        self,
        molecular=Mol["methane"],
        name="methane",
        basis="sto-3g",
        if_basis_str=False,
        spin=0,
    ):
        self.name = name
        self.basis = basis
        self.if_basis_str = if_basis_str

        self.mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                self.basis,
                self.if_basis_str,
            ),
            verbose=4,
            spin=spin,
        )
        print(self.mol.atom)

    def mrks_diis(
        self,
        frac_old,
        load_inv,
        diis_n=15,
        vxc_inv=None,
        max_inv_step=2500,
        cc_triple=False,
    ):
        """
        Generate 1-RDM.
        """
        print(f"Mrks diis module. Generate {self.name}")
        return mrks_diis(
            self,
            frac_old,
            load_inv,
            diis_n,
            vxc_inv=vxc_inv,
            max_inv_step=max_inv_step,
            cc_triple=cc_triple,
        )

    def umrks_diis(
        self,
        frac_old,
        load_inv,
        diis_n=15,
        vxc_inv=None,
        max_inv_step=2500,
        cc_triple=False,
    ):
        """
        Generate 1-RDM.
        """
        print(f"Umrks diis module. Generate {self.name}")
        return umrks_diis(
            self,
            frac_old,
            load_inv,
            diis_n=diis_n,
            vxc_inv=vxc_inv,
            max_inv_step=max_inv_step,
            cc_triple=cc_triple,
        )

    def gmrks_diis(self, frac_old, load_inv):
        """
        Generate 1-RDM.
        """
        print(f"Umrks diis module. Generate {self.name}")
        gmrks_diis(self, frac_old, load_inv)

    # pylint: disable=W0201
    def test_mol(self, require_grad=False):
        """
        Generate 1-RDM.
        """
        self.grids = Grid(self.mol)
        self.ao_0 = pyscf.dft.numint.eval_ao(self.mol, self.grids.coords)
        self.ao_1 = pyscf.dft.numint.eval_ao(self.mol, self.grids.coords, deriv=1)
        self.grids_test = Grid(self.mol, level=3, period=2)
        self.ao_0_test = pyscf.dft.numint.eval_ao(self.mol, self.grids_test.coords)

        print(f"Generate data for {self.name}")

        if self.mol.spin == 0:
            time_start = timer()
            mdft = pyscf.scf.RKS(self.mol)
            mdft.xc = "b3lyp"
            mdft.max_cycle = 250
            mdft.kernel()
            self.dm1_dft = mdft.make_rdm1(ao_repr=True)
            self.e_dft = mdft.e_tot
            self.dft_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=self.dm1_dft,
                unit="A.U.",
            )
            if require_grad:
                g = mdft.nuc_grad_method()
                self.grad_dft = g.kernel()
            else:
                self.grad_dft = None
            self.time_dft = timer() - time_start

            time_start = timer()
            mf = pyscf.scf.RHF(self.mol)
            mf.kernel()
            mycc = pyscf.cc.CCSD(mf)
            mycc.incore_complete = True
            mycc.async_io = False
            mycc.direct = True
            mycc.kernel()
            self.dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
            self.cc_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=self.dm1_cc,
                unit="A.U.",
            )
            if require_grad:
                g = ccsd_grad.Gradients(mycc)
                self.grad_ccsd = g.kernel()
            else:
                self.grad_ccsd = None
            self.time_cc = timer() - time_start
        else:
            time_start = timer()
            mdft = pyscf.scf.UKS(self.mol)
            mdft.xc = "b3lyp"
            mdft.max_cycle = 250
            mdft.kernel()
            self.dm1_dft = mdft.make_rdm1(ao_repr=True)
            self.e_dft = mdft.e_tot
            self.dft_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=self.dm1_dft,
                unit="A.U.",
            )
            if require_grad:
                g = mdft.nuc_grad_method()
                self.grad_dft = g.kernel()
            else:
                self.grad_dft = None
            self.time_dft = timer() - time_start

            time_start = timer()
            mf = pyscf.scf.UHF(self.mol)
            mf.kernel()
            mycc = pyscf.cc.UCCSD(mf)
            mycc.incore_complete = True
            mycc.async_io = False
            mycc.direct = True
            mycc.kernel()
            self.dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
            self.cc_dipole = pyscf.scf.uhf.dip_moment(
                mol=self.mol,
                dm=self.dm1_cc,
                unit="A.U.",
            )
            if require_grad:
                g = uccsd_grad.Gradients(mycc)
                self.grad_ccsd = g.kernel()
            else:
                self.grad_ccsd = None
            self.time_cc = timer() - time_start
