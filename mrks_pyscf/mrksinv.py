"""@package docstring
Documentation for this module.
 
More details.
"""

import json
import gc
from functools import partial
from pathlib import Path
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt

import opt_einsum as oe
from scipy import linalg as LA

import pyscf
from pyscf import dft
from pyscf.dft import numint as ni

from .utils.grids import Grid
from .utils.aux_function import Auxfunction
from .utils.kernel import kernel
from .utils.gen_tau_rho import gen_taup_rho, gen_tau_rho
from .utils.gen_w import gen_w_vec
from .utils.mol import BASIS, BASISTRAN

DIRPATH = Path(__file__).resolve().parents[0]


@dataclass
class Args:
    """
    This class is used to store the arguments.
    """

    level: int
    inv_step: int
    scf_step: int
    device: str
    noisy_print: bool
    basis: str
    if_basis_str: bool
    error_inv: float
    error_scf: float
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
        basis="sto-3g",
        inv_step=25000,
        scf_step=2500,
        device=None,
        noisy_print=False,
        if_basis_str=False,
        error_inv=1e-6,
        error_scf=1e-8,
    ):
        if args is None:
            self.args = Args(
                level,
                inv_step,
                scf_step,
                device,
                noisy_print,
                basis,
                if_basis_str,
                error_inv,
                error_scf,
                frac_old,
            )
        else:
            self.args = Args(
                args.level,
                args.inv_step,
                args.scf_step,
                args.device,
                args.noisy_print,
                BASISTRAN[args.basis] if args.basis in BASISTRAN else args.basis,
                args.if_basis_str,
                args.error_inv,
                args.error_scf,
                frac_old,
            )

        if self.args.device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(self.args.device)

        self.au2kjmol = 2625.5
        self.path = path
        # make directory if not exist
        if not self.path.exists():
            self.path.mkdir(parents=True)

        basis = {}
        for i_atom in molecular:
            if self.args.if_basis_str:
                import basis_set_exchange

                basis[i_atom[0]] = pyscf.gto.load(
                    (
                        basis_set_exchange.api.get_basis(
                            BASIS[self.args.basis.lower()],
                            elements=i_atom[0],
                            fmt="nwchem",
                        )
                        if ((i_atom[0] == "H") and (self.args.basis.lower() in BASIS))
                        else basis_set_exchange.api.get_basis(
                            self.args.basis.lower(), elements=i_atom[0], fmt="nwchem"
                        )
                    ),
                    i_atom[0],
                )
            else:
                basis[i_atom[0]] = (
                    BASIS[self.args.basis.lower()]
                    if ((i_atom[0] == "H") and (self.args.basis.lower() in BASIS))
                    else self.args.basis.lower()
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

        self.myhf = pyscf.scf.RHF(self.mol)
        self.myhf.kernel()
        self.norb = self.myhf.mo_energy.shape[0]
        self.mo = self.myhf.mo_coeff
        self.logger.info(f"Number of orbital: {self.norb} \n")

        self.grids = Grid(self.mol, self.args.level)
        ao_value = ni.eval_ao(self.mol, self.grids.coords, deriv=1)
        self.ao_0 = ni.eval_ao(self.mol, self.grids.coords, deriv=0)
        self.ao_1 = ni.eval_ao(self.mol, self.grids.coords, deriv=1)[1:]
        ao_2 = ni.eval_ao(self.mol, self.grids.coords, deriv=2)
        self.ao_2_diag = ao_2[4, :, :] + ao_2[7, :, :] + ao_2[9, :, :]
        self.eval_rho = partial(ni.eval_rho, self.mol, ao_value)

        self.nuc = self.mol.intor("int1e_nuc")
        self.kin = self.mol.intor("int1e_kin")
        self.h1e = self.nuc + self.kin
        self.h1_mo = np.einsum("ab,ai,bj->ij", self.h1e, self.mo, self.mo)
        self.eri = self.mol.intor("int2e")
        self.eri_mo = pyscf.ao2mo.kernel(self.eri, self.mo, compact=False)
        self.aux_function = Auxfunction(self)

        self.error_vxc = 0
        self.error_dm1 = 0
        self.potential_shift = 0

        self.mdft = self.mol.KS()
        self.mdft.xc = "b3lyp"
        self.mdft.kernel()
        self.logger.info(f"\nxc type: {self.mdft.xc}\n")
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

    def save_mol_info(self):
        """
        This function is used to save the molecular information.
        """
        np.save(self.path / "saved_data" / "coordinate.npy", self.grids.coords)
        save_json = {}
        save_json["atom"] = self.mol.atom
        save_json["basis"] = self.mol.basis
        with open(self.path / "mol_info.json", "w", encoding="utf-8") as f:
            json.dump(save_json, f, indent=4)

    def save_b3lyp(self):
        """
        Save the b3lyp exchange correlation energy and potential.
        Only LDA level can be used.
        """
        if self.dm1_inv is None:
            dm1_inv_grid = np.load(self.path / "rho_inv_mrks.npy")
            dm1_inv_r = self.grids.matrix_to_vector(dm1_inv_grid)
        else:
            dm1_inv_r = self.aux_function.oe_rho_r(self.dm1_inv * 2, backend="torch")
        exc, vxc = dft.libxc.eval_xc("SVWN", dm1_inv_r)[:2]
        vxc = vxc[0]

        vxc_mrks_grid = self.grids.vector_to_matrix(vxc)
        exc_mrks_grid = self.grids.vector_to_matrix(exc)
        vxc_mrks_check = self.grids.matrix_to_vector(vxc_mrks_grid)
        exc_mrks_check = self.grids.matrix_to_vector(exc_mrks_grid)
        self.logger.info(
            f"{np.linalg.norm(vxc - vxc_mrks_check):16.10f}\n"
            f"{np.linalg.norm(exc - exc_mrks_check):16.10f}\n"
        )
        np.save(self.path / "lda.npy", vxc_mrks_grid)
        np.save(self.path / "lda_e.npy", exc_mrks_grid)

        # self.mdft.xc = "b3lyp"
        # dm1 = self.mdft.make_rdm1()
        # ao_value = ni.eval_ao(self.mol, self.grids.coords, deriv=1)
        # rho = ni.eval_rho(self.mol, ao_value, dm1, xctype="MGGA")
        # exc, vxc = dft.libxc.eval_xc(self.mdft.xc, rho)[:2]
        # vxc = vxc[0]
        # xc_v = self.aux_function.oe_fock(vxc, self.grids.weights)
        # # exc +=
        # self.logger.info(f"\nExc = {np.sum(exc * rho[0] * self.grids.weights)}\n")

        # e_nuc = oe.contract("ij,ji->", self.h1e, dm1)
        # e_vj = oe.contract("pqrs,pq,rs->", self.eri, dm1, dm1)
        # e_vk = oe.contract("pqrs,pr,qs->", self.eri, dm1, dm1)
        # ene_t_vc = (
        #     e_nuc
        #     + self.mol.energy_nuc()
        #     + e_vj * 0.5
        #     - e_vk * 0.05
        #     + (exc * rho[0] * self.grids.weights).sum()
        # )
        # self.logger.info(f"Ene_t_vc = {ene_t_vc}\n")
        # self.logger.info(f"Total energy = {self.mdft.e_tot}\n")
        # self.logger.info(f"Shape of vxc: {np.shape(vxc)}\n")
        # self.logger.info(f"Shape of weights: {np.shape(self.grids.weights)}\n")
        # self.logger.info(
        #     f"Before scf: {np.array2string(dm1, precision=4, separator=',', suppress_small=True)}\n"
        # )

        # for step in range(200):
        #     rho = ni.eval_rho(self.mol, ao_value, dm1, xctype="MGGA")
        #     exc, vxc = dft.libxc.eval_xc(self.mdft.xc, rho)[:2]
        #     rho_1 = 2 * np.einsum("uv, rgu, gv -> rg", dm1, self.ao_1, self.ao_0)

        #     xc_v = 0.5 * self.aux_function.oe_fock(vxc[0], self.grids.weights)
        #     xc_v += 2 * oe.contract(
        #         "g, g, rg, rgu, gv -> uv",
        #         vxc[1],
        #         self.grids.weights,
        #         rho_1,
        #         self.ao_1,
        #         self.ao_0,
        #     )
        #     xc_v = xc_v + xc_v.T
        #     vjk = self.myhf.get_jk(self.mol, dm1, 1)

        #     fock_a = self.mat_hs @ (self.h1e + vjk[0] + xc_v) @ self.mat_hs
        #     _, mo = np.linalg.eigh(fock_a)
        #     mo = self.mat_hs @ mo
        #     dm1_old = dm1.copy()
        #     dm1 = 2 * mo[:, : self.nocc] @ mo[:, : self.nocc].T
        #     error = np.linalg.norm(dm1 - dm1_old)
        #     dm1 = self.hybrid(dm1, dm1_old)

        # self.logger.info(f"step: {step:<8} error of dm1, {error:.2e}\n")
        # self.logger.info(
        #     f"After scf: {np.array2string(dm1, precision=4, separator=',', suppress_small=True)}\n"
        # )

    def hybrid(self, new, old):
        """
        Generate the hybrid density matrix.
        """
        return new * (1 - self.args.frac_old) + old * self.args.frac_old

    def kernel(self, method="fci", gen_dm2=True):
        """
        This function is used to do the quantum chemistry calculation using pyscf.
        """
        if ((self.dm1 is not None)) and ((self.dm2 is not None)):
            self.logger.info("dm1 and dm2 are already calculated.\n")
        else:
            self.e, self.dm1_mo, self.dm2_mo, if_mo = kernel(method, self.myhf, gen_dm2)
            self.logger.info("dm1_mo, dm2_mo done.\n")

            if if_mo:
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
            else:
                # fall back to the AO basis. dm1_mo is dm1_ao.
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

            self.logger.info("dm1 dm2 done.\n")
            self.logger.info(
                f"Total energy: {self.e:16.10f}\n" f"The dm1 and dm2 are generated.\n"
            )
        self.gen_e_vxc()
        self.gen_e_bar_wf()

    def save_kernel_dm12(self):
        """
        Do NOT use this function. Cost too much disk space (40Gb * 40 = 1.6T).
        """

    def load_kernel_dm12(self):
        """
        Do NOT use this function. Cost too much disk space.
        """

    def gen_e_vxc(self):
        """
        This function is used to generate the exchange-correlation energy on the grid.
        """
        dm1_r = self.aux_function.oe_rho_r(self.dm1)
        dm1_cuda = torch.from_numpy(self.dm1).to(self.device)
        self.exc_over_dm = np.zeros(len(self.grids.coords))

        # 2rdm
        dm2_cuda = torch.from_numpy(self.dm2).to(self.device)
        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            dm2_cuda,
            (self.norb,),
            (self.norb,),
            (self.norb, self.norb),
            constants=[0],
            optimize="optimal",
        )

        for i, coord in enumerate(self.grids.coords):
            if i % 1000 == 0:
                self.logger.info(f"\n2Rdm, Grid {i:<8} of {len(self.grids.coords):<8}")
            elif i % 100 == 0:
                self.logger.info(".")

            ao_0_i = torch.from_numpy(self.ao_0[i]).to(self.device)

            with self.mol.with_rinv_origin(coord):
                rinv = self.mol.intor("int1e_rinv")
                rinv = torch.from_numpy(rinv).to(self.device)
                self.exc_over_dm[i] += (
                    expr_rinv_dm2_r(ao_0_i, ao_0_i, rinv, backend="torch") / 2
                )

        del dm2_cuda
        torch.cuda.empty_cache()
        self.logger.info(f"\nAfter 2Rdm,\n {torch.cuda.memory_summary()}.\n\n")
        self.exc_over_dm = self.exc_over_dm / dm1_r

        # 1rdm
        expr_rinv_dm1_r = oe.contract_expression(
            "ij,ij->",
            dm1_cuda,
            (self.norb, self.norb),
            constants=[0],
            optimize="optimal",
        )

        self.logger.info(f"\nAfter 1Rdm,\n {torch.cuda.memory_summary()}.\n\n")
        for i, coord in enumerate(self.grids.coords):
            if i % 1000 == 0:
                self.logger.info(f"\n1Rdm, Grid {i:<8} of {len(self.grids.coords):<8}")
            elif i % 100 == 0:
                self.logger.info(".")

            with self.mol.with_rinv_origin(coord):
                rinv = self.mol.intor("int1e_rinv")
                rinv = torch.from_numpy(rinv).to(self.device)
                rinv_dm1_r = expr_rinv_dm1_r(rinv, backend="torch")
                self.exc_over_dm[i] += -rinv_dm1_r / 2

        self.exc = self.exc_over_dm * dm1_r
        ene_vc = np.sum(self.exc * self.grids.weights)
        error = ene_vc - (
            np.einsum("pqrs,pqrs", self.eri, self.dm2).real / 2
            - np.einsum("pqrs,pq,rs", self.eri, self.dm1, self.dm1).real / 2
        )
        self.logger.info(
            f"\nenergy: {ene_vc:<10.4e}, error {error:16.10f}\n"
            f"The exchange-correlation energy is generated.\n\n"
        )

        self.logger.info(f"\nSummary of Exc, \n {torch.cuda.memory_summary()}.\n\n")
        del self.dm2
        gc.collect()
        torch.cuda.empty_cache()

    def gen_e_bar_wf(self):
        """
        This function is used to prepare the inverse calculation.
        """
        generalized_fock = self.dm1_mo @ self.h1_mo + oe.contract(
            "rsnq,rsmq->mn", self.eri_mo, self.dm2_mo
        )
        dm1_r = self.aux_function.oe_rho_r(self.dm1.copy() / 2, backend="torch")
        del self.dm2_mo, self.h1_mo
        gc.collect()
        torch.cuda.empty_cache()

        generalized_fock = 0.5 * (generalized_fock + generalized_fock.T)
        eig_e, eig_v = np.linalg.eigh(generalized_fock)
        eig_v = self.mo @ eig_v
        eig_e = eig_e / 2
        e_bar_r_wf = (
            self.aux_function.oe_ebar_r_wf(eig_e, eig_v, eig_v, backend="torch") / dm1_r
        )
        self.logger.info("E_bar_r_wf done.\n")

        self.emax = np.max(e_bar_r_wf)
        self.v_vxc_e_taup = -e_bar_r_wf

    def save_prepare_inverse(self):
        """
        Do NOT use this function all the time. Cost too much disk space.
        """
        if not (self.path / "saved_data").exists():
            (self.path / "saved_data").mkdir(parents=True)
        np.save(self.path / "saved_data" / "v_vxc_e_taup.npy", self.v_vxc_e_taup)
        np.save(self.path / "saved_data" / "exc_over_dm.npy", self.exc_over_dm)
        np.save(self.path / "saved_data" / "exc.npy", self.exc)
        np.save(self.path / "saved_data" / "dm1.npy", self.dm1)
        np.save(self.path / "saved_data" / "dm1_mo.npy", self.dm1_mo)
        np.save(self.path / "saved_data" / "mo.npy", self.mo)
        np.save(self.path / "saved_data" / "emax.npy", self.emax)
        np.save(self.path / "saved_data" / "e.npy", self.e)

    def load_prepare_inverse(self):
        """
        Do NOT use this function all the time.
        """
        self.v_vxc_e_taup = np.load(self.path / "saved_data" / "v_vxc_e_taup.npy")
        self.exc_over_dm = np.load(self.path / "saved_data" / "exc_over_dm.npy")
        self.exc = np.load(self.path / "saved_data" / "exc.npy")
        self.dm1 = np.load(self.path / "saved_data" / "dm1.npy")
        self.dm1_mo = np.load(self.path / "saved_data" / "dm1_mo.npy")
        self.mo = np.load(self.path / "saved_data" / "mo.npy")
        self.emax = np.load(self.path / "saved_data" / "emax.npy")
        self.e = np.load(self.path / "saved_data" / "e.npy")

    def inv(self):
        """
        This function is used to do the total inverse calculation.
        """
        self.gen_taup_rho_wf()
        self.inverse()
        self.check()
        self.save_data()

    def gen_taup_rho_wf(self):
        """
        This function is used to generate the tau_rho_wf.
        """
        dm1_r = self.aux_function.oe_rho_r(self.dm1.copy() / 2, backend="torch")
        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(self.dm1_mo)
        eigs_v_dm1 = self.mo @ eigs_v_dm1
        eigs_e_dm1 = eigs_e_dm1 / 2
        eigs_v_dm1_cuda = torch.from_numpy(eigs_v_dm1).to(self.device)

        self.taup_rho_wf = gen_taup_rho(
            self.aux_function,
            dm1_r,
            eigs_v_dm1_cuda,
            eigs_e_dm1,
            backend="torch",
            logger=self.logger,
        )
        self.tau_rho_wf = gen_tau_rho(
            self.aux_function,
            dm1_r,
            eigs_v_dm1_cuda,
            eigs_e_dm1,
            backend="torch",
            logger=self.logger,
        )
        self.logger.info("\nTaup_rho done.\n")

        self.logger.info(
            f"\nSummary of prepare_inverse, \n {torch.cuda.memory_summary()}.\n\n"
        )
        del self.dm1_mo
        torch.cuda.empty_cache()

    def inverse(self):
        """
        This function is used to do the inverse calculation.
        """
        mo = self.mo.copy()
        eigvecs = self.myhf.mo_energy.copy()
        self.dm1_inv = self.dm1.copy() / 2
        self.vj = self.myhf.get_jk(self.mol, self.dm1_inv * 2, 1)[0]
        self.vxc = np.zeros_like(self.v_vxc_e_taup)
        dm1_r = self.aux_function.oe_rho_r(self.dm1.copy() / 2, backend="torch")
        self.v_vxc_e_taup += self.exc_over_dm * 2 + self.taup_rho_wf / dm1_r
        self.vxc = dft.libxc.eval_xc(
            "B88,P86", self.eval_rho(self.dm1_inv * 2, xctype="GGA")
        )[1][0]
        print(np.shape(self.vxc))

        for i in range(self.args.inv_step):
            self.vj = self.hybrid(
                self.myhf.get_jk(self.mol, 2 * self.dm1_inv, 1)[0], self.vj
            )
            dm1_inv_r = self.aux_function.oe_rho_r(self.dm1_inv, backend="torch")
            self.potential_shift = self.emax - np.max(eigvecs[: self.nocc])
            eigvecs_cuda = torch.from_numpy(eigvecs).to(self.device)
            mo = torch.from_numpy(mo).to(self.device)

            ebar_ks = self.aux_function.oe_ebar_r_ks(
                eigvecs_cuda[: self.nocc] + self.potential_shift,
                mo[:, : self.nocc],
                mo[:, : self.nocc],
                backend="torch",
            )
            ebar_ks = ebar_ks.cpu().numpy() / dm1_inv_r

            self.taup_rho_ks = gen_taup_rho(
                self.aux_function,
                dm1_inv_r,
                mo[:, : self.nocc],
                np.ones(self.nocc),
                backend="torch",
            )

            vxc_old = self.vxc.copy()
            self.vxc = self.v_vxc_e_taup + ebar_ks - self.taup_rho_ks / dm1_inv_r
            self.error_vxc = np.linalg.norm((self.vxc - vxc_old) * self.grids.weights)
            self.vxc = self.hybrid(self.vxc, vxc_old)

            xc_v = self.aux_function.oe_fock(
                self.vxc, self.grids.weights, backend="torch"
            )
            eigvecs, mo = np.linalg.eigh(
                self.mat_hs @ (self.h1e + self.vj + xc_v) @ self.mat_hs
            )
            mo = self.mat_hs @ mo
            dm1_inv_old = self.dm1_inv.copy()
            self.dm1_inv = mo[:, : self.nocc] @ mo[:, : self.nocc].T
            self.error_dm1 = np.linalg.norm(self.dm1_inv - dm1_inv_old)

            if self.args.noisy_print:
                self.logger.info(
                    "\n%s %s %s %s ",
                    f"step:{i:<8}",
                    f"error of vxc: {self.error_vxc::<10.5e}",
                    f"dm: {self.error_dm1::<10.5e}",
                    f"shift: {self.potential_shift::<10.5e}",
                )
            else:
                if i % 100 == 0:
                    self.logger.info(
                        "\n%s %s %s %s ",
                        f"step:{i:<8}",
                        f"error of vxc: {self.error_vxc::<10.5e}",
                        f"dm: {self.error_dm1::<10.5e}",
                        f"shift: {self.potential_shift::<10.5e}",
                    )
                elif i % 10 == 0:
                    self.logger.info(".")

            if (i > 0) and (self.error_vxc < self.args.error_inv):
                break

        mo = torch.from_numpy(mo).to(self.device)
        self.tau_rho_ks = gen_tau_rho(
            self.aux_function,
            dm1_inv_r,
            mo[:, : self.nocc],
            np.ones(self.nocc),
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
        dm1_dft = self.mdft.make_rdm1()
        dm1_dft_r = self.aux_function.oe_rho_r(dm1_dft, backend="torch")
        dm1_scf = self.scf(dm1_dft)
        dm1_scf_r = self.aux_function.oe_rho_r(dm1_scf, backend="torch")

        kin_correct = 2 * np.sum(
            (self.tau_rho_wf - self.tau_rho_ks) * self.grids.weights
        )

        kin_correct1 = 2 * np.sum(
            (self.taup_rho_wf - self.taup_rho_ks) * self.grids.weights
        )

        save_data = {}
        save_data["energy_dft"] = self.au2kjmol * (self.mdft.e_tot - self.e)
        save_data["energy"] = self.au2kjmol * self.e
        save_data["correct kinetic energy"] = kin_correct * self.au2kjmol
        save_data["correct kinetic energy1"] = kin_correct1 * self.au2kjmol

        error_inv_r = np.sum(np.abs(dm1_inv_r - dm1_r) * self.grids.weights)
        error_scf_r = np.sum(np.abs(dm1_scf_r - dm1_r) * self.grids.weights)
        error_dft_r = np.sum(np.abs(dm1_dft_r - dm1_r) * self.grids.weights)
        save_data["error of dm1_inv"] = error_inv_r
        save_data["error of dm1_scf"] = error_scf_r
        save_data["error of dm1_dft"] = error_dft_r

        self.logger.info(
            f"\nCheck! {error_inv_r:16.10f}\n"
            f"{error_scf_r:16.10f}\n"
            f"{error_dft_r:16.10f}\n"
        )

        save_data["energy_inv"] = self.au2kjmol * (
            self.gen_energy(dm1_inv, kin_correct=kin_correct) - self.e
        )
        save_data["energy_scf"] = self.au2kjmol * (
            self.gen_energy(dm1_scf, kin_correct=kin_correct) - self.e
        )
        save_data["energy_inv1"] = self.au2kjmol * (
            self.gen_energy(dm1_inv, kin_correct=kin_correct1) - self.e
        )
        save_data["energy_scf1"] = self.au2kjmol * (
            self.gen_energy(dm1_scf, kin_correct=kin_correct1) - self.e
        )
        save_data["energy_exa"] = self.au2kjmol * (self.gen_energy(self.dm1) - self.e)

        save_data["energy_exa_w"] = self.au2kjmol * (
            self.gen_energy_w(dm1, dm1_r, kin_correct) - self.e
        )
        save_data["energy_inv_w"] = self.au2kjmol * (
            self.gen_energy_w(dm1_inv, dm1_inv_r, kin_correct, True) - self.e
        )
        save_data["energy_scf_w"] = self.au2kjmol * (
            self.gen_energy_w(dm1_scf, dm1_scf_r, kin_correct, True) - self.e
        )
        save_data["energy_exa_w1"] = self.au2kjmol * (
            self.gen_energy_w(dm1, dm1_r, kin_correct1) - self.e
        )
        save_data["energy_inv_w1"] = self.au2kjmol * (
            self.gen_energy_w(dm1_inv, dm1_inv_r, kin_correct1, True) - self.e
        )
        save_data["energy_scf_w1"] = self.au2kjmol * (
            self.gen_energy_w(dm1_scf, dm1_scf_r, kin_correct1, True) - self.e
        )

        with open(self.path / "save_data.json", "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4)

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
        tr_grid = self.grids.vector_to_matrix(2 * (self.tau_rho_wf - self.tau_rho_ks))
        trp_grid = self.grids.vector_to_matrix(
            2 * (self.taup_rho_wf - self.taup_rho_ks)
        )
        weight_grid = self.grids.vector_to_matrix(self.grids.weights)

        rho_0_check = self.grids.matrix_to_vector(rho_0_grid)
        rho_t_check = self.grids.matrix_to_vector(rho_t_grid)
        vxc_mrks_check = self.grids.matrix_to_vector(vxc_mrks_grid)
        exc_mrks_check = self.grids.matrix_to_vector(exc_mrks_grid)
        exc_dm_check = self.grids.matrix_to_vector(exc_dm_grid)
        tr_check = self.grids.matrix_to_vector(tr_grid)
        trp_check = self.grids.matrix_to_vector(trp_grid)
        weight_check = self.grids.matrix_to_vector(weight_grid)

        self.logger.info(
            f"\nCheck! {np.linalg.norm(rho_0 - rho_0_check):16.10f}\n"
            f"{np.linalg.norm(rho_t - rho_t_check):16.10f}\n"
            f"{np.linalg.norm(self.vxc - vxc_mrks_check):16.10f}\n"
            f"{np.linalg.norm(self.exc - exc_mrks_check):16.10f}\n"
            f"{np.linalg.norm(self.exc_over_dm - exc_dm_check):16.10f}\n"
            f"{np.linalg.norm(2 * (self.tau_rho_wf - self.tau_rho_ks) - tr_check):16.10f}\n"
            f"{np.linalg.norm(2 * (self.taup_rho_wf - self.taup_rho_ks) - trp_check):16.10f}\n"
            f"{np.linalg.norm(weight_check - self.grids.weights):16.10f}\n"
        )

        np.save(self.path / "mrks.npy", vxc_mrks_grid)
        np.save(self.path / "mrks_e.npy", exc_mrks_grid)
        np.save(self.path / "mrks_e_dm.npy", exc_dm_grid)
        np.save(self.path / "rho_inv_mrks.npy", rho_0_grid)
        np.save(self.path / "rho_t_mrks.npy", rho_t_grid)
        np.save(self.path / "tr.npy", tr_grid)
        np.save(self.path / "trp.npy", trp_grid)
        np.save(self.path / "weight.npy", weight_grid)

        f, axes = plt.subplots(self.mol.natm, 5)
        for i in range(self.mol.natm):
            axes[i, 0].imshow(-rho_t_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 1].imshow(vxc_mrks_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 2].imshow(exc_mrks_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 3].imshow(exc_dm_grid[0, :, :], cmap="Greys", aspect="auto")
            axes[i, 4].imshow(tr_grid[0, :, :], cmap="Greys", aspect="auto")
        plt.savefig(self.path / "fig.pdf")

    def scf(self, dm1):
        """
        This function is used to do the SCF calculation.
        """
        vxc = self.vxc
        xc_v = self.aux_function.oe_fock(vxc, self.grids.weights, backend="torch")

        for step in range(self.args.scf_step):
            vj = self.myhf.get_jk(self.mol, dm1, 1)[0]
            fock_a = self.mat_hs @ (self.h1e + vj + xc_v) @ self.mat_hs
            _, mo = np.linalg.eigh(fock_a)
            mo = self.mat_hs @ mo
            dm1_old = dm1.copy()
            dm1 = 2 * mo[:, : self.nocc] @ mo[:, : self.nocc].T
            error = np.linalg.norm(dm1 - dm1_old)
            dm1 = self.hybrid(dm1, dm1_old)
            if error < self.args.error_scf:
                self.logger.info(f"error of dm1 in the last step, {error:.2e}\n")
                break
            if step % 100 == 0:
                self.logger.info(f"step: {step:<8} error of dm1, {error:.2e}\n")
        return dm1

    def gen_energy(
        self,
        dm1,
        kin_correct=None,
    ):
        """
        This function is used to check the energy.
        """
        e_h1 = oe.contract("ij,ji->", self.h1e, dm1)
        e_vj = oe.contract("pqrs,pq,rs->", self.eri, dm1, dm1)

        ene_t_vc = (
            e_h1
            + self.mol.energy_nuc()
            + e_vj * 0.5
            + np.sum(self.exc * self.grids.weights)
        )

        if kin_correct is not None:
            ene_t_vc += kin_correct

        return ene_t_vc

    def gen_energy_w(self, dm1, dm1_r, kin_correct, kin_corr=False):
        """
        This function is used to check the energy, via the W function.
        """
        w_vec = gen_w_vec(
            dm1,
            dm1_r,
            self.ao_0,
            self.ao_1,
            self.vxc,
            self.grids.coords,
        )
        e_nuc = oe.contract("ij,ji->", self.h1e, dm1)
        e_vj = oe.contract("pqrs,pq,rs->", self.eri, dm1, dm1)
        if kin_corr:
            ene_t_vc = (
                e_nuc
                + self.mol.energy_nuc()
                + e_vj * 0.5
                + (w_vec * self.grids.weights).sum()
                - kin_correct
            )
        else:
            ene_t_vc = (
                e_nuc
                + self.mol.energy_nuc()
                + e_vj * 0.5
                + (w_vec * self.grids.weights).sum()
                - 2 * kin_correct
            )
        return ene_t_vc
