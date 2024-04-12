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
from .utils.gen_tau_rho import gen_tau_rho
from .utils.kernel import kernel
from .utils.rotate import rotate
from .utils.mol import BASIS, BASISTRAN

DIRPATH = Path(__file__).resolve().parents[0]


@dataclass
class Args:
    """
    This class is used to store the arguments.
    """

    level: int
    device: str
    noisy_print: bool
    basis: str
    if_basis_str: bool


class DFT2CC:
    """Documentation for a class."""

    def __init__(
        self,
        molecular,
        path=Path(__file__).resolve().parents[0],
        args=None,
        logger=None,
        level=3,
        basis="sto-3g",
        device=None,
        noisy_print=False,
        if_basis_str=False,
    ):
        if args is None:
            self.args = Args(
                level,
                device,
                noisy_print,
                basis,
                if_basis_str,
            )
        else:
            self.args = Args(
                args.level,
                args.device,
                args.noisy_print,
                BASISTRAN[args.basis] if args.basis in BASISTRAN else args.basis,
                args.if_basis_str,
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
        rotate(molecular)

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
        self.logger.info(f"Level of grid: {self.args.level} \n")
        self.logger.info(f"Basis set: {self.args.basis.lower()} \n")
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

        self.exc = None
        self.tau_rho_wf = None
        self.ene_nuc = None

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
        dm1_r = self.aux_function.oe_rho_r(self.dm1.copy() / 2, backend="torch")
        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(self.dm1_mo)
        eigs_v_dm1 = self.mo @ eigs_v_dm1
        eigs_e_dm1 = eigs_e_dm1 / 2
        eigs_v_dm1_cuda = torch.from_numpy(eigs_v_dm1).to(self.device)
        self.tau_rho_wf = 2 * gen_tau_rho(
            self.aux_function,
            dm1_r,
            eigs_v_dm1_cuda,
            eigs_e_dm1,
            backend="torch",
            logger=self.logger,
        )

        self.ene_nuc = np.zeros(len(self.grids.coords))
        for i, coord in enumerate(self.grids.coords):
            if i % 10000 == 0:
                self.logger.info(f"\nNuc, Grid {i:<8} of {len(self.grids.coords):<8}")
            elif i % 1000 == 0:
                self.logger.info(".")
            for i_atom in range(self.mol.natm):
                self.ene_nuc[i] += (
                    2
                    * dm1_r[i]
                    * self.mol.atom_charges()[i_atom]
                    / np.linalg.norm(self.mol.atom_coords()[i_atom], coord)
                )

        self.exc = np.zeros(len(self.grids.coords))
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
                int1e_rinv = self.mol.intor("int1e_rinv")
                int1e_rinv = torch.from_numpy(int1e_rinv).to(self.device)
                self.exc[i] += (
                    expr_rinv_dm2_r(ao_0_i, ao_0_i, int1e_rinv, backend="torch") / 2
                )

        del dm2_cuda
        torch.cuda.empty_cache()
        self.logger.info(f"\nAfter 2Rdm,\n {torch.cuda.memory_summary()}.\n\n")

        ene_vc = np.sum(self.exc * self.grids.weights)
        kin = np.sum(self.tau_rho_wf * self.grids.weights)
        nuc = np.sum(self.ene_nuc * self.grids.weights)
        error = (
            ene_vc
            + kin
            + nuc
            - np.einsum("ij,ji->", self.h1e, self.dm1)
            - (np.einsum("pqrs,pqrs", self.eri, self.dm2).real / 2)
        )
        self.logger.info(
            f"\nenergy: {ene_vc:<10.4e}, error {error:16.10f}\n"
            f"The exchange-correlation energy is generated.\n\n"
        )

        self.logger.info(f"\nSummary of Exc, \n {torch.cuda.memory_summary()}.\n\n")
        del self.dm2
        gc.collect()
        torch.cuda.empty_cache()

    def check(self):
        """
        This function is used to check the density matrix and the energy.
        """
        self.logger.info(
            "Check the density matrix and the energy.\n"
            "WARNING!!!!\n"
            "WARNING!!!!\n"
            "WARNING!!!!\n"
            "This part of code should not be used in the production.\n"
        )
        save_data = {}
        save_data["energy_dft"] = self.au2kjmol * (self.mdft.e_tot - self.e)
        save_data["energy"] = self.au2kjmol * self.e
        save_data["error of energy"] = self.au2kjmol * (
            self.gen_energy(self.dm1) - self.e
        )
        # rho_t = self.aux_function.oe_rho_r(self.dm1, backend="torch")
        # save_data["error of energy"] = self.au2kjmol * (self.gen_energy_rho(rho_t) - self.e)

        with open(self.path / "save_data.json", "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4)

    def save_data(self):
        """
        This function is used to save the training data.
        """
        rho_t = self.aux_function.oe_rho_r(self.dm1, backend="torch")
        dm1_dft = self.mdft.make_rdm1()
        rho_dft = self.aux_function.oe_rho_r(dm1_dft, backend="torch")

        rho_t_grid = self.grids.vector_to_matrix(rho_t)
        rho_dft_grid = self.grids.vector_to_matrix(rho_dft)
        exc_mrks_grid = self.grids.vector_to_matrix(self.exc)
        weight_grid = self.grids.vector_to_matrix(self.grids.weights)

        rho_t_check = self.grids.matrix_to_vector(rho_t_grid)
        rho_dft_check = self.grids.matrix_to_vector(rho_dft_grid)
        exc_mrks_check = self.grids.matrix_to_vector(exc_mrks_grid)
        weight_check = self.grids.matrix_to_vector(weight_grid)

        self.logger.info(
            f"{np.linalg.norm(rho_t - rho_t_check):16.10f}\n"
            f"{np.linalg.norm(rho_dft - rho_dft_check):16.10f}\n"
            f"{np.linalg.norm(self.exc - exc_mrks_check):16.10f}\n"
            f"{np.linalg.norm(weight_check - self.grids.weights):16.10f}\n"
        )

        np.save(self.path / "e_output.npy", exc_mrks_grid)
        np.save(self.path / "rho_output.npy", rho_t_grid)
        np.save(self.path / "rho_input.npy", rho_dft_grid)
        np.save(self.path / "weight.npy", weight_grid)

    def gen_energy(
        self,
        dm1,
    ):
        """
        This function is used to check the energy.
        """
        e_h1 = oe.contract("ij,ji->", self.nuc, dm1)

        ene_t_vc = (
            e_h1
            + self.mol.energy_nuc()
            + np.sum((self.tau_rho_wf + self.exc) * self.grids.weights)
        )

        return ene_t_vc
