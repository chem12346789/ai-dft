"""@package docstring
Documentation for this module.
 
More details.
"""

from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import opt_einsum as oe
from scipy import linalg as LA

import pyscf
from pyscf import dft

import psi4

from src.mrks_pyscf.utils.grids import Grid
from src.mrks_pyscf.utils.aux_function import Auxfunction
from src.mrks_pyscf.utils.kernel import kernel
from src.mrks_pyscf.utils.gen_taup_rho import gen_taup_rho, gen_tau_rho
from src.mrks_pyscf.utils.gen_w import gen_w_vec
from src.mrks_pyscf.utils.mol import BASIS, BASIS_PSI4


@dataclass
class Args:
    level: int
    inv_step: int
    scf_step: int
    device: str
    noisy_print: bool
    psi4: bool
    basis: str
    frac_old: float


class Mrksinv:
    """Documentation for a class."""

    def __init__(
        self,
        molecular,
        path=Path(__file__).resolve().parents[0],
        args=None,
        logger=None,
        frac_old=0.8,
        level=3,
        inv_step=25000,
        scf_step=2500,
        device=None,
        noisy_print=False,
        if_psi4=False,
        basis="sto-3g",
    ):
        if args is None:
            self.args = Args(
                level, inv_step, scf_step, device, noisy_print, if_psi4, basis, frac_old
            )
        else:
            self.args = Args(
                args.level,
                args.inv_step,
                args.scf_step,
                args.device,
                args.noisy_print,
                args.psi4,
                args.basis,
                frac_old,
            )
            self.args = args
        if self.args.device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)
        self.au2kjmol = 2625.5
        self.path = path
        # make directory if not exist
        if not self.path.exists():
            self.path.mkdir(parents=True)

        if self.args.psi4:
            basis = {}
            for i_atom in molecular:
                basis[i_atom[0]] = BASIS_PSI4[self.args.basis][i_atom[0]]
        else:
            basis = {}
            for i_atom in molecular:
                basis[i_atom[0]] = (
                    BASIS[self.args.basis]
                    if ((i_atom[0] == "H") and (self.args.basis in BASIS))
                    else self.args.basis
                )

        mol = pyscf.M(
            atom=molecular,
            basis=basis,
        )

        self.mol = mol
        self.mat_s = mol.intor("int1e_ovlp")
        self.mat_hs = LA.fractional_matrix_power(self.mat_s, -0.5).real
        self.nocc = mol.nelec[0]

        self.logger = logger
        self.logger.info(f"Path: {self.path} \n")
        self.logger.info(f"Device: {self.device} \n")
        self.logger.info(f"Fraction of old: {self.args.frac_old} \n")
        self.logger.info(f"Level of grid: {self.args.level} \n")
        self.logger.info(f"Basis set: {self.mol.basis} \n")
        self.logger.info(f"Unit of distance: {self.mol.unit} \n")
        self.logger.info(f"Info of molecule: {self.mol.atom} \n")

        self.mol.verbose = 0
        self.mol.output = self.path / "pyscf.log"

        self.myhf = pyscf.scf.HF(self.mol)
        self.myhf.kernel()
        self.norb = self.myhf.mo_energy.shape[0]
        self.mo = self.myhf.mo_coeff

        self.grids = Grid(self.mol, self.args.level)
        ni = dft.numint.NumInt()
        self.ao_0 = ni.eval_ao(self.mol, self.grids.coords, deriv=0)
        self.ao_1 = ni.eval_ao(self.mol, self.grids.coords, deriv=1)[1:]
        ao_2 = ni.eval_ao(self.mol, self.grids.coords, deriv=2)
        self.ao_2_diag = ao_2[4, :, :] + ao_2[7, :, :] + ao_2[9, :, :]

        self.nuc = self.mol.intor("int1e_nuc")
        self.kin = self.mol.intor("int1e_kin")
        self.h1e = self.nuc + self.kin
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
        self.exc_over_dm = None
        self.v_vxc_e_taup = None
        self.taup_rho_wf = None
        self.taup_rho_ks = None
        self.tau_rho_wf = None
        self.tau_rho_ks = None
        self.emax = None

        self.dm1_inv = None
        self.eigs_e_dm1 = None
        self.eigs_v_dm1 = None

    def kernel(self, method="fci", gen_dm2=True):
        """
        This function is used to do the quantum chemistry calculation.
        """
        if self.args.psi4:
            self.kernel_psi4(method, gen_dm2=gen_dm2)
        else:
            self.kernel_pyscf(method, gen_dm2=gen_dm2)

    def kernel_psi4(self, method="fci", gen_dm2=True):
        """
        This function is used to do the quantum chemistry calculation using psi4.
        """
        mol_str = ""
        for atom in self.mol.atom:
            mol_str += f"{atom[0]} {atom[1]} {atom[2]} {atom[3]}\n"
        mol_str += "noreorient\n"
        mol_str += "nocom\n"
        mol_str += "units angstrom\n"
        mol_str += "symmetry c1\n"
        mol = psi4.geometry(mol_str)

        psi4.set_output_file("output.dat", True)
        psi4.core.set_num_threads(12)
        psi4.core.clean()
        psi4.set_options(
            {
                "reference": "rhf",
                "opdm": True,
                "tpdm": True,
            }
        )

        self.e, wfn = psi4.energy(
            f"{method}/{self.args.basis}", return_wfn=True, molecule=mol
        )
        if method == "hf":
            self.logger.info("HF method.\n")
            self.dm1 = wfn.Da().np + wfn.Db().np
            self.dm1_mo = oe.contract(
                "ij,pi,qj->pq",
                self.dm1,
                (self.mo).T @ self.mat_s,
                (self.mo).T @ self.mat_s,
            )
            if gen_dm2:
                self.dm2 = (
                    np.einsum("ij,kl->ijkl", self.dm1, self.dm1)
                    - np.einsum("ij,kl->iklj", self.dm1, self.dm1) / 2
                )
                self.dm2_mo = oe.contract(
                    "pqrs,ip,jq,ur,vs->ijuv",
                    self.dm2,
                    (self.mo).T @ self.mat_s,
                    (self.mo).T @ self.mat_s,
                    (self.mo).T @ self.mat_s,
                    (self.mo).T @ self.mat_s,
                )
        else:
            self.logger.info("CI method.\n")
            self.dm1_mo = wfn.get_opdm(-1, -1, "SUM", True).np
            self.dm1 = oe.contract("ij,pi,qj->pq", self.dm1_mo, self.mo, self.mo)

            if gen_dm2:
                # obtain the memory of 2-RDM
                self.logger.info(
                    f"memory of 2-RDM: {(self.dm1.shape[0] ** 4) * 8.e-9 * 2} GB\n"
                )
                self.logger.info(
                    f"Total energy: {self.e:16.10f}\n"
                    f"The dm1 and dm2 are generated.\n",
                )
                self.dm2_mo = wfn.get_tpdm("SUM", True).np
                self.dm2 = oe.contract(
                    "pqrs,ip,jq,ur,vs->ijuv",
                    self.dm2_mo,
                    self.mo,
                    self.mo,
                    self.mo,
                    self.mo,
                )
        self.vj = self.myhf.get_jk(self.mol, self.dm1, 1)[0]

    def kernel_pyscf(self, method="fci", gen_dm2=True):
        """
        This function is used to do the quantum chemistry calculation using pyscf.
        """
        if ((self.dm1 is not None)) and ((self.dm2 is not None)):
            self.logger.info("dm1 and dm2 are already calculated.\n")
        else:
            self.e, self.dm1_mo, self.dm2_mo = kernel(method, self.myhf, gen_dm2)
            self.logger.info("dm1_mo, dm2_mo done.\n")

            if method == "hf":
                self.dm1 = self.dm1_mo.copy()
                if gen_dm2:
                    self.dm2 = self.dm2_mo.copy()
                self.dm1_mo = oe.contract(
                    "ij,pi,qj->pq",
                    self.dm1,
                    (self.mo).T @ self.mat_s,
                    (self.mo).T @ self.mat_s,
                )
                if gen_dm2:
                    self.dm2_mo = oe.contract(
                        "pqrs,ip,jq,ur,vs->ijuv",
                        self.dm2,
                        (self.mo).T @ self.mat_s,
                        (self.mo).T @ self.mat_s,
                        (self.mo).T @ self.mat_s,
                        (self.mo).T @ self.mat_s,
                    )

            else:
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
                f"Total energy: {self.e:16.10f}\n" f"The dm1 and dm2 are generated.\n"
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
            f"\nenergy: {ene_vc:<10.4e}, error {error:16.10f}\n"
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

        self.tau_rho_wf = gen_tau_rho(
            self.aux_function.oe_tau_rho,
            dm1_r,
            eigs_v_dm1_cuda,
            eigs_e_dm1 * 2,
            backend="torch",
            logger=self.logger,
        )
        self.logger.info("\nTau_rho done.\n")

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

        dm1_r = self.aux_function.oe_rho_r(self.dm1_inv, backend="torch")
        cut_off_r = np.ones_like(dm1_r)
        cut_off_r[dm1_r < 1e-10] = 0
        self.exc_over_dm = (self.exc + 1e-14) / (dm1_r + 1e-14) * cut_off_r
        self.v_vxc_e_taup += self.exc_over_dm

        for i in range(self.args.inv_step):
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
                np.ones(self.nocc),
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
                        f"error of vxc: {error_vxc::<10.5e}",
                        f"dm: {error_dm1::<10.5e}",
                        f"shift: {potential_shift::<10.5e}",
                    )
                else:
                    if i % 100 == 0:
                        self.logger.info(
                            "\n%s %s %s %s ",
                            f"step:{i:<8}",
                            f"error of vxc: {error_vxc::<10.5e}",
                            f"dm: {error_dm1::<10.5e}",
                            f"shift: {potential_shift::<10.5e}",
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
            fock_a = self.mat_hs @ (self.h1e + self.vj + xc_v) @ self.mat_hs
            eigvecs, mo = np.linalg.eigh(fock_a)
            mo = self.mat_hs @ mo
            dm1_inv_old = self.dm1_inv.copy()
            self.dm1_inv = mo[:, : self.nocc] @ mo[:, : self.nocc].T

        dm1_inv_r = self.aux_function.oe_rho_r(self.dm1_inv, backend="torch")
        if mo is np.ndarray:
            mo = torch.from_numpy(mo).to(self.device)
        self.tau_rho_ks = gen_tau_rho(
            self.aux_function.oe_tau_rho,
            dm1_inv_r,
            mo[:, : self.nocc],
            np.ones(self.nocc) * 2,
            backend="torch",
        )
        self.logger.info("\nTau_rho_ks done.\n")
        self.logger.info("\ninverse done.\n\n")

    def check(self):
        """
        This function is used to check the density matrix and the energy.
        """
        dm1_inv = self.dm1_inv * 2
        dm1_inv_r = self.aux_function.oe_rho_r(dm1_inv, backend="torch")
        dm1 = self.dm1.copy()
        dm1_r = self.aux_function.oe_rho_r(self.dm1, backend="torch")

        mdft = self.mol.KS()
        mdft.xc = "b3lyp"
        mdft.kernel()
        dm1_dft = mdft.make_rdm1()
        dm1_scf = self.scf(dm1_dft)
        dm1_scf_r = self.aux_function.oe_rho_r(dm1_scf, backend="torch")

        error_inv_r = np.sum(np.abs(dm1_inv_r - dm1_r) * self.grids.weights)
        error_scf_r = np.sum(np.abs(dm1_scf_r - dm1_r) * self.grids.weights)
        self.logger.info(f"\nerror of dm1_inv, {error_inv_r:16.10f}")
        self.logger.info(f"\nerror of dm1_scf, {error_scf_r:16.10f}")

        self.logger.info(f"\nuse the w function to check the energy.\n")

        w_vec = gen_w_vec(
            dm1,
            dm1_r,
            self.ao_0,
            self.ao_1,
            self.vxc,
            self.grids.coords,
        )
        e_nuc = oe.contract("ij,ji->", self.h1e, self.dm1)
        e_vj = oe.contract("pqrs,pq,rs->", self.eri, self.dm1, self.dm1)
        ene_t_vc = (
            e_nuc
            + self.mol.energy_nuc()
            + e_vj * 0.5
            + (w_vec * self.grids.weights).sum()
            - 2 * ((self.tau_rho_wf - self.tau_rho_ks) * self.grids.weights).sum()
        )
        self.logger.info(
            f"\nexact energy: {((ene_t_vc - self.e) * self.au2kjmol):16.10f} kj/mol\n"
        )

        w_vec_inv = gen_w_vec(
            dm1_inv,
            dm1_inv_r,
            self.ao_0,
            self.ao_1,
            self.vxc,
            self.grids.coords,
        )
        e_nuc_inv = oe.contract("ij,ji->", self.h1e, dm1_inv)
        e_vj_inv = oe.contract("pqrs,pq,rs->", self.eri, dm1_inv, dm1_inv)
        ene_0_vc_inv = (
            e_nuc_inv
            + self.mol.energy_nuc()
            + e_vj_inv * 0.5
            + (w_vec_inv * self.grids.weights).sum()
            - ((self.tau_rho_wf - self.tau_rho_ks) * self.grids.weights).sum()
        )
        self.logger.info(
            f"inverse energy: {((ene_0_vc_inv - self.e) * self.au2kjmol):16.10f} kj/mol\n"
        )

        w_vec_scf = gen_w_vec(
            dm1_scf,
            dm1_scf_r,
            self.ao_0,
            self.ao_1,
            self.vxc,
            self.grids.coords,
        )
        e_nuc_scf = oe.contract("ij,ji->", self.h1e, dm1_scf)
        e_vj_scf = oe.contract("pqrs,pq,rs->", self.eri, dm1_scf, dm1_scf)
        ene_0_vc_scf = (
            e_nuc_scf
            + self.mol.energy_nuc()
            + e_vj_scf * 0.5
            + (w_vec_scf * self.grids.weights).sum()
            - ((self.tau_rho_wf - self.tau_rho_ks) * self.grids.weights).sum()
        )
        self.logger.info(
            f"scf energy: {((ene_0_vc_scf - self.e) * self.au2kjmol):16.10f} kj/mol\n"
        )

        self.logger.info(
            "%s",
            f"energy_inv: {2625.5 * (self.gen_energy(dm1_inv, if_kin_correct=True) - self.e):16.10f}\n",
        )
        self.logger.info(
            "%s",
            f"energy_scf: {2625.5 * (self.gen_energy(dm1_scf, if_kin_correct=True) - self.e):16.10f}\n",
        )
        self.logger.info(
            "%s",
            f"energy_exa: {2625.5 * (self.gen_energy(self.dm1) - self.e):16.10f}\n",
        )
        self.logger.info("%s", f"ene_dft: {2625.5 * (mdft.e_tot - self.e):16.10f}\n")
        self.logger.info("%s", f"ene_exa: {2625.5 * self.e:16.10f}\n")

        self.logger.info(
            f"correct kinetic energy: {(((self.tau_rho_wf - self.tau_rho_ks) * self.grids.weights).sum() * self.au2kjmol):16.10f} kj/mol\n"
        )

        self.logger.info(
            f"error: {(np.sum((w_vec - 2 * (self.tau_rho_wf - self.tau_rho_ks) - self.exc) * self.grids.weights) * self.au2kjmol):16.10f} kj/mol\n"
        )

    def save_data(self):
        """
        This function is used to save the training data.
        """
        rho_0 = self.aux_function.oe_rho_r(self.dm1_inv * 2)
        rho_0_grid = self.grids.vector_to_matrix(rho_0)
        rho_t = self.aux_function.oe_rho_r(self.dm1)
        rho_t_grid = self.grids.vector_to_matrix(rho_t)

        vxc_mrks_grid = self.grids.vector_to_matrix(self.vxc)
        exc_mrks_grid = self.grids.vector_to_matrix(self.exc)
        exc_dm_grid = self.grids.vector_to_matrix(self.exc_over_dm)
        tr_grid = self.grids.vector_to_matrix(self.tau_rho_wf - self.tau_rho_ks)

        rho_0_check = self.grids.matrix_to_vector(rho_0_grid)
        rho_t_check = self.grids.matrix_to_vector(rho_t_grid)
        vxc_mrks_check = self.grids.matrix_to_vector(vxc_mrks_grid)
        exc_mrks_check = self.grids.matrix_to_vector(exc_mrks_grid)
        exc_dm_check = self.grids.matrix_to_vector(exc_dm_grid)
        tr_check = self.grids.matrix_to_vector(tr_grid)

        self.logger.info(
            f"\nCheck! {np.linalg.norm(rho_0 - rho_0_check):16.10f}\n"
            f"{np.linalg.norm(rho_t - rho_t_check):16.10f}\n"
            f"{np.linalg.norm(self.vxc - vxc_mrks_check):16.10f}\n"
            f"{np.linalg.norm(self.exc - exc_mrks_check):16.10f}\n"
            f"{np.linalg.norm(self.exc_over_dm - exc_dm_check):16.10f}\n"
            f"{np.linalg.norm(self.tau_rho_wf - self.tau_rho_ks - tr_check):16.10f}\n"
        )

        np.save(self.path / "mrks.npy", vxc_mrks_grid)
        np.save(self.path / "mrks_e.npy", exc_mrks_grid)
        np.save(self.path / "mrks_e_dm.npy", exc_dm_grid)
        np.save(self.path / "rho_inv_mrks.npy", rho_0_grid)
        np.save(self.path / "rho_t_mrks.npy", rho_t_grid)
        np.save(self.path / "tr.npy", tr_grid)

        f, axes = plt.subplots(self.mol.natm, 5)
        for i in range(self.mol.natm):
            axes[i, 0].imshow(-rho_t_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 1].imshow(vxc_mrks_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 2].imshow(exc_mrks_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 3].imshow(exc_dm_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 3].imshow(tr_grid[0, :, :], cmap="Greys", aspect="auto")
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
            fock_a = self.mat_hs @ (self.h1e + vj + xc_v) @ self.mat_hs
            _, mo = np.linalg.eigh(fock_a)
            mo = self.mat_hs @ mo
            dm1_old = dm1.copy()
            dm1 = 2 * mo[:, : self.nocc] @ mo[:, : self.nocc].T
            error = np.linalg.norm(dm1 - dm1_old)
            if (error < 1e-8) or (step > self.args.scf_step):
                self.logger.info(f"error of dm1 in the last step, {error:.2e}\n")
                flag = False
            else:
                if step % 100 == 0:
                    self.logger.info(f"step: {step:<8} error of dm1, {error:.2e}\n")
            dm1 = dm1 * (1 - self.frac_old) + dm1_old * self.frac_old
        return dm1

    def gen_energy(
        self,
        dm1,
        if_kin_correct=False,
    ):
        """
        This function is used to check the energy.
        """
        rho_0 = np.einsum("uv, gu, gv -> g", dm1, self.ao_0, self.ao_0)
        e_h1 = oe.contract("ij,ji->", self.h1e, dm1)
        e_vj = oe.contract("pqrs,pq,rs->", self.eri, self.dm1, self.dm1)

        ene_t_vc = (
            e_h1
            + self.mol.energy_nuc()
            + e_vj * 0.5
            + np.sum((self.exc_over_dm * rho_0 / 2) * self.grids.weights)
        )

        if if_kin_correct:
            ene_t_vc += np.sum((self.tau_rho_wf - self.tau_rho_ks) * self.grids.weights)

        return ene_t_vc
