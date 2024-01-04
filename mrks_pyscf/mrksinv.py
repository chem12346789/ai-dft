"""@package docstring
Documentation for this module.
 
More details.
"""

from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

import opt_einsum as oe
from scipy import linalg as LA

import pyscf
from pyscf import dft

from .utils.grids import Grid
from .utils.aux_function import Auxfunction
from .utils.kernel import kernel
from .utils.gen_taup_rho import gen_taup_rho


class Mrksinv:
    """Documentation for a class."""

    def __init__(
        self,
        mol,
        frac_old=0.8,
        level=3,
        inv_step=25000,
        scf_step=2500,
        path=Path(__file__).resolve().parents[0],
        logger=None,
        device=None,
        noisy_print=False,
    ):
        self.mol = mol
        s_0_ao = mol.intor("int1e_ovlp")
        self.mats = LA.fractional_matrix_power(s_0_ao, -0.5).real
        self.nocc = mol.nelec[0]
        self.frac_old = frac_old
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)
        self.noise_print = noisy_print

        self.path = path
        # make directory if not exist
        if not self.path.exists():
            self.path.mkdir(parents=True)
        self.inv_step = inv_step
        self.scf_step = scf_step

        self.logger = logger
        self.logger.info(f"Path: {self.path} \n")
        self.logger.info(f"Device: {self.device} \n")
        self.logger.info(f"Fraction of old: {self.frac_old} \n")
        self.logger.info(f"Level of grid: {level} \n")
        self.logger.info(f"Basis set: {self.mol.basis} \n")
        self.logger.info(f"Unit of distance: {self.mol.unit} \n")
        self.logger.info(f"Info of molecule: {self.mol.atom} \n")

        self.mol.verbose = 0
        self.mol.output = self.path / "pyscf.log"

        self.myhf = pyscf.scf.HF(self.mol)
        self.myhf.kernel()
        self.norb = self.myhf.mo_energy.shape[0]
        self.mo = self.myhf.mo_coeff

        self.grids = Grid(self.mol, level)
        ni = dft.numint.NumInt()
        self.ao_0 = ni.eval_ao(self.mol, self.grids.coords, deriv=0)
        self.ao_1 = ni.eval_ao(self.mol, self.grids.coords, deriv=1)[1:]
        ao_2 = ni.eval_ao(self.mol, self.grids.coords, deriv=2)
        self.ao_2_diag = ao_2[4, :, :] + ao_2[7, :, :] + ao_2[9, :, :]

        self.nuc = self.mol.intor("int1e_nuc")
        self.h1e = self.nuc + self.mol.intor("int1e_kin")
        self.h1_mo = np.einsum("ab,ai,bj->ij", self.h1e, self.mo, self.mo)
        self.eri = self.mol.intor("int2e")
        self.eri_mo = pyscf.ao2mo.kernel(self.eri, self.mo, compact=False)
        self.aux_function = Auxfunction(self)

        shapes = np.shape(self.h1e), np.shape(self.h1e), np.shape(self.h1e)
        self.oe_dm1_ao = oe.contract_expression("ij,pi,qj->pq", *shapes)

        self.dm1 = None
        self.dm2 = None
        self.dm1_mo = None
        self.dm2_mo = None
        self.e = None

        self.vj = None
        self.exc = None
        self.vxc = None
        self.exc_kin_correct = None
        self.v_vxc_e_taup = None
        self.taup_rho_wf = None
        self.taup_rho_ks = None
        self.emax = None

        self.dm1_inv = None
        self.eigs_e_dm1 = None
        self.eigs_v_dm1 = None

    def kernel(self, method="fci", gen_dm2=True):
        """
        This function is used to do the quantum chemistry calculation.
        """
        if ((self.dm1 is not None)) and ((self.dm2 is not None)):
            self.logger.info("dm1 and dm2 are already calculated.\n")
        else:
            self.e, self.dm1_mo, self.dm2_mo = kernel(method, self.myhf, gen_dm2)
            self.logger.info("dm1_mo, dm2_mo done.\n")
            self.dm1 = oe.contract("ij,pi,qj->pq", self.dm1_mo, self.mo, self.mo)
            if gen_dm2:
                self.dm2 = oe.contract(
                    "pqrs,ip,jq,ur,vs->ijuv",
                    self.dm2_mo,
                    self.mo,
                    self.mo,
                    self.mo,
                    self.mo,
                )
            self.logger.info("dm1 dm2 done.\n")
            self.vj = self.myhf.get_jk(self.mol, self.dm1, 1)[0]
            self.logger.info(
                f"Total energy: {self.e:<10.2e}\n" f"The dm1 and dm2 are generated.\n"
            )

    def save_kernel_dm12(self):
        """
        Do NOT use this function. Cost too much disk space.
        """

    def load_kernel_dm12(self):
        """
        Do NOT use this function.
        """

    def inv_prepare(self):
        """
        This function is used to do the total inverse calculation.
        Note all the data will be transform to the torch tensor.
        """
        self.gen_e_vxc()
        self.prepare_inverse()

    def inv(self):
        """
        This function is used to do the total inverse calculation.
        """
        self.inverse()
        self.check()
        self.save_data()

    def gen_e_vxc(self):
        """
        This function is used to generate the exchange-correlation energy on the grid.
        """
        dm1_r = self.aux_function.oe_rho_r(self.dm1)

        # 1rdm
        dm1_cuda = torch.from_numpy(self.dm1).to(self.device)

        expr_rinv_dm1_r = oe.contract_expression(
            "ij,ij->",
            dm1_cuda,
            (self.norb, self.norb),
            constants=[0],
            optimize="optimal",
        )

        self.exc = np.zeros(len(self.grids.coords))
        self.logger.info(f"\nBefore 1Rdm,\n {torch.cuda.memory_summary()}.\n\n")
        for i, coord in enumerate(self.grids.coords):
            if i % 100 == 0:
                self.logger.info(f"\n1Rdm, Grid {i:<8} of {len(self.grids.coords):<8}")
            elif i % 10 == 0:
                self.logger.info(".")

            with self.mol.with_rinv_origin(coord):
                rinv = self.mol.intor("int1e_rinv")
                rinv = torch.from_numpy(rinv).to(self.device)
                rinv_dm1_r = expr_rinv_dm1_r(rinv, backend="torch")
                self.exc[i] = -rinv_dm1_r * dm1_r[i] / 2

        del dm1_cuda
        torch.cuda.empty_cache()
        self.logger.info(f"\nBefore 2Rdm,\n {torch.cuda.memory_summary()}.\n\n")

        # 2rdm
        dm2_cuda = torch.from_numpy(self.dm2).to(self.device)
        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,kl,i,j->",
            dm2_cuda,
            (self.norb, self.norb),
            (self.norb,),
            (self.norb,),
            constants=[0],
            optimize="optimal",
        )

        for i, coord in enumerate(self.grids.coords):
            if i % 100 == 0:
                self.logger.info(f"\n2Rdm, Grid {i:<8} of {len(self.grids.coords):<8}")
            elif i % 10 == 0:
                self.logger.info(".")
            ao_0_i = torch.from_numpy(self.ao_0[i]).to(self.device)

            with self.mol.with_rinv_origin(coord):
                rinv = self.mol.intor("int1e_rinv")
                rinv = torch.from_numpy(rinv).to(self.device)
                rinv_dm2_r = expr_rinv_dm2_r(rinv, ao_0_i, ao_0_i, backend="torch")
                self.exc[i] += rinv_dm2_r / 2

        ene_vc = (self.exc * self.grids.weights).sum()
        error = ene_vc - (
            np.einsum("pqrs,pqrs", self.eri, self.dm2).real / 2
            - np.einsum("pqrs,pq,rs", self.eri, self.dm1, self.dm1).real / 2
        )
        self.logger.info(
            f"\nenergy: {ene_vc:<10.4e}, error {error:<10.2e}\n"
            f"The exchange-correlation energy is generated.\n\n"
        )

        self.logger.info(f"\nSummary of Exc, \n {torch.cuda.memory_summary()}.\n\n")
        del self.dm2, dm2_cuda
        torch.cuda.empty_cache()

    def prepare_inverse(self):
        """
        This function is used to prepare the inverse calculation.
        """
        generalized_fock = self.dm1_mo @ self.h1_mo + oe.contract(
            "rsnq,rsmq->mn", self.eri_mo, self.dm2_mo
        )
        dm1_inv = self.dm1.copy() / 2
        dm1_r = self.aux_function.oe_rho_r(dm1_inv, backend="torch")

        generalized_fock = 0.5 * (generalized_fock + generalized_fock.T)
        eig_e, eig_v = np.linalg.eigh(generalized_fock)
        eig_v = self.mo @ eig_v
        eig_e = eig_e / 2
        e_bar_r_wf = (
            self.aux_function.oe_ebar_r_wf(eig_e, eig_v, eig_v, backend="torch") / dm1_r
        )
        self.logger.info("E_bar_r_wf done.\n")

        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(self.dm1_mo)
        eigs_v_dm1 = self.mo @ eigs_v_dm1
        eigs_e_dm1 = eigs_e_dm1 / 2
        eigs_v_dm1_cuda = torch.from_numpy(eigs_v_dm1).to(self.device)

        self.taup_rho_wf = gen_taup_rho(
            self.aux_function.oe_taup_rho,
            dm1_r,
            eigs_v_dm1_cuda,
            eigs_e_dm1,
            backend="torch",
            logger=self.logger,
        )
        self.logger.info("\nTaup_rho done.\n")

        self.emax = np.max(e_bar_r_wf)
        self.v_vxc_e_taup = -e_bar_r_wf + self.taup_rho_wf

        self.logger.info(
            f"\nSummary of prepare_inverse, \n {torch.cuda.memory_summary()}.\n\n"
        )
        del self.dm1_mo, self.h1_mo
        torch.cuda.empty_cache()

    def inverse(self):
        """
        This function is used to do the inverse calculation.
        """
        mo = self.mo.copy()
        eigvecs = self.myhf.mo_energy.copy()
        self.dm1_inv = self.dm1.copy() / 2
        self.v_vxc_e_taup += self.exc / self.aux_function.oe_rho_r(
            self.dm1_inv, backend="torch"
        )

        for i in range(self.inv_step):
            dm1_inv_r = self.aux_function.oe_rho_r(self.dm1_inv, backend="torch")
            potential_shift = self.emax - np.max(eigvecs[: self.nocc])
            eigvecs_cuda = torch.from_numpy(eigvecs).to(self.device)
            mo = torch.from_numpy(mo).to(self.device)

            ebar_ks = self.aux_function.oe_ebar_r_ks(
                eigvecs_cuda[: self.nocc] + potential_shift,
                mo[:, : self.nocc],
                mo[:, : self.nocc],
                backend="torch",
            )
            ebar_ks = ebar_ks.cpu().numpy() / dm1_inv_r

            self.taup_rho_ks = gen_taup_rho(
                self.aux_function.oe_taup_rho,
                dm1_inv_r,
                mo[:, : self.nocc],
                np.ones_like(eigvecs[: self.nocc]),
                backend="torch",
            )

            self.vxc = self.v_vxc_e_taup + ebar_ks - self.taup_rho_ks
            if i > 0:
                error_vxc = np.linalg.norm((self.vxc - vxc_old) * self.grids.weights)
                error_dm1 = np.linalg.norm(self.dm1_inv - dm1_inv_old)
                if self.noise_print:
                    self.logger.info(
                        "\n%s %s %s %s ",
                        f"step:{i:<8}",
                        f"error of vxc: {error_vxc:<10.2e}",
                        f"dm: {error_dm1:<10.2e}",
                        f"shift: {potential_shift:<10.2e}",
                    )
                else:
                    if i % 100 == 0:
                        self.logger.info(
                            "\n%s %s %s %s ",
                            f"step:{i:<8}",
                            f"error of vxc: {error_vxc:<10.2e}",
                            f"dm: {error_dm1:<10.2e}",
                            f"shift: {potential_shift:<10.2e}",
                        )
                    elif i % 10 == 0:
                        self.logger.info(".")

                self.vxc = self.vxc * (1 - self.frac_old) + vxc_old * self.frac_old
                if error_vxc < 1e-6:
                    break
            else:
                self.logger.info(f"Begin inverse calculation. step: {i:<38} ")

            vxc_old = self.vxc.copy()
            xc_v = self.aux_function.oe_fock(
                self.vxc, self.grids.weights, backend="torch"
            )
            fock_a = self.mats @ (self.h1e + self.vj + xc_v) @ self.mats
            eigvecs, mo = np.linalg.eigh(fock_a)
            mo = self.mats @ mo
            dm1_inv_old = self.dm1_inv.copy()
            self.dm1_inv = mo[:, : self.nocc] @ mo[:, : self.nocc].T
        self.logger.info("\ninverse done.\n\n")

    def check(self):
        """
        This function is used to check the density matrix and the energy.
        """
        dm1_inv = self.dm1_inv * 2
        dm1_inv_r = self.aux_function.oe_rho_r(dm1_inv, backend="torch")
        dm1_r = self.aux_function.oe_rho_r(self.dm1, backend="torch")

        error_inv_r = np.sum(np.abs(dm1_inv_r - dm1_r) * self.grids.weights)
        self.logger.info(f"\nerror of dm1_inv, {error_inv_r:<10.2e}")

        e_nuc = oe.contract("ij,ji->", self.nuc, self.dm1)
        e_vj = oe.contract("pqrs,pq,rs->", self.eri, self.dm1, self.dm1)

        au2kjmol = 2625.5
        t_r = -0.5 * np.einsum("uv, gu, gv -> g", self.dm1, self.ao_0, self.ao_2_diag)
        self.exc_kin_correct = self.exc + t_r

        ene_t_vc = (
            e_nuc
            + self.mol.energy_nuc()
            + e_vj * 0.5
            + (self.exc_kin_correct * self.grids.weights).sum()
        )

        self.logger.info(
            f"\nerror of energy: {((ene_t_vc - self.e) * au2kjmol):<10.2e} kj/mol\n"
        )

    def save_data(self):
        """
        This function is used to save the training data.
        """
        rho_0 = np.einsum("uv, gu, gv -> g", self.dm1_inv * 2, self.ao_0, self.ao_0)
        rho_0_grid = self.grids.vector_to_matrix(rho_0)
        rho_t = np.einsum("uv, gu, gv -> g", self.dm1, self.ao_0, self.ao_0)
        rho_t_grid = self.grids.vector_to_matrix(rho_t)

        cut_off_r = np.ones_like(rho_t)
        cut_off_r[rho_t < 1e-10] = 0
        exc_over_dm = (self.exc_kin_correct + 1e-14) / (rho_t + 1e-14) * cut_off_r

        vxc_mrks_grid = self.grids.vector_to_matrix(self.vxc)
        exc_mrks_grid = self.grids.vector_to_matrix(self.exc_kin_correct)
        exc_dm_grid = self.grids.vector_to_matrix(exc_over_dm)

        rho_0_check = self.grids.matrix_to_vector(rho_0_grid)
        rho_t_check = self.grids.matrix_to_vector(rho_t_grid)
        vxc_mrks_check = self.grids.matrix_to_vector(vxc_mrks_grid)
        exc_mrks_check = self.grids.matrix_to_vector(exc_mrks_grid)
        exc_dm_check = self.grids.matrix_to_vector(exc_dm_grid)

        self.logger.info(
            f"\nCheck! {np.linalg.norm(rho_0 - rho_0_check):<10.2e}\n"
            f"{np.linalg.norm(rho_t - rho_t_check):<10.2e}\n"
            f"{np.linalg.norm(self.vxc - vxc_mrks_check):<10.2e}\n"
            f"{np.linalg.norm(self.exc_kin_correct - exc_mrks_check):<10.2e}\n"
            f"{np.linalg.norm(exc_over_dm - exc_dm_check):<10.2e}\n"
        )

        np.save(self.path / "mrks.npy", vxc_mrks_grid)
        np.save(self.path / "mrks_e.npy", exc_mrks_grid)
        np.save(self.path / "mrks_e_dm.npy", exc_dm_grid)
        np.save(self.path / "rho_scf_mrks.npy", rho_0_grid)
        np.save(self.path / "rho_t_mrks.npy", rho_t_grid)
        np.save(self.path / "dm1_inv.npy", self.dm1_inv)

        f, axes = plt.subplots(self.mol.natm, 4)
        for i in range(self.mol.natm):
            axes[i, 0].imshow(-rho_t_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 1].imshow(vxc_mrks_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 2].imshow(exc_mrks_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 3].imshow(exc_dm_grid[0, :, :], cmap="Greys", aspect="auto")
        plt.savefig(self.path / "fig.pdf")

    def scf(self, dm1):
        """
        This function is used to do the SCF calculation.
        """
        vxc = self.vxc
        xc_v = self.aux_function.oe_fock(vxc, self.grids.weights, backend="torch")

        flag = True
        step = 0
        while flag:
            step += 1
            vj = self.myhf.get_jk(self.mol, dm1, 1)[0]
            fock_a = self.mats @ (self.h1e + vj + xc_v) @ self.mats
            _, mo = np.linalg.eigh(fock_a)
            mo = self.mats @ mo
            dm1_old = dm1.copy()
            dm1 = 2 * mo[:, : self.nocc] @ mo[:, : self.nocc].T
            error = np.linalg.norm(dm1 - dm1_old)
            if (error < 1e-10) or (step > self.scf_step):
                self.logger.info(f"error of dm1 in the last step, {error:.2e}")
                flag = False
            else:
                if step % 100 == 0:
                    self.logger.info(f"step: {step:<8} error of dm1, {error:.2e}")
            dm1 = dm1 * (1 - self.frac_old) + dm1_old * self.frac_old
        return dm1

    def gen_energy(self, dm1, exc_kin_correct=None):
        """
        This function is used to check the energy.
        """
        if exc_kin_correct is None:
            exc_kin_correct = self.exc_kin_correct
        nuc = self.mol.intor("int1e_nuc")
        e_nuc = oe.contract("ij,ji->", nuc, dm1)
        e_vj = oe.contract("pqrs,pq,rs->", self.eri, self.dm1, self.dm1)

        ene_t_vc = (
            e_nuc
            + self.mol.energy_nuc()
            + e_vj * 0.5
            + (self.exc_kin_correct * self.grids.weights).sum()
        )

        return ene_t_vc
