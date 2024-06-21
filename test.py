"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
import copy
from itertools import product
from pathlib import Path

import pyscf
import torch
import numpy as np
import opt_einsum as oe
from torch.utils.data import DataLoader

from cadft import CC_DFT_DATA, add_args, gen_logger
from cadft.utils import Mol
from cadft.utils import ModelDict


AU2KCALMOL = 627.5096080306


def process(data, dtype):
    """
    Load the whole data to the gpu.
    """
    if len(data.shape) == 4:
        return data.to(
            device="cuda",
            dtype=dtype,
            memory_format=torch.channels_last,
        )
    else:
        return data.to(
            device="cuda",
            dtype=dtype,
        )


class BasicDataset:
    """
    Documentation for a class.
    """

    def __init__(self, input_, weight_, batch_size, dtype):
        self.input = input_
        self.weight = weight_
        self.ids = list(input_.keys())
        self.batch_size = batch_size
        if dtype == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input": self.input[self.ids[idx]],
            "weight": self.weight[self.ids[idx]],
        }

    def load_to_gpu(self):
        """
        Load the whole data to the device.
        """
        dataloader = DataLoader(
            self,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
        )

        dataloader_gpu = []
        for batch in dataloader:
            batch_gpu = {}
            # move images and labels to correct device and type
            (
                batch_gpu["input"],
                batch_gpu["weight"],
            ) = (
                process(batch["input"], self.dtype),
                process(batch["weight"], self.dtype),
            )
            dataloader_gpu.append(batch_gpu)
        return dataloader_gpu


if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init the model
    modeldict = ModelDict(
        args.hidden_size,
        args.num_layers,
        args.residual,
        device,
        args.precision,
        if_mkdir=False,
    )
    modeldict.load_model(args.load)

    # 2. Test loop
    distance_l = gen_logger(args.distance_list)
    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        args.name_mol,
        args.extend_atom,
        args.extend_xyz,
        distance_l,
    ):
        # 2.0 Prepare
        molecular = copy.deepcopy(Mol[name_mol])
        print(f"Generate {name_mol}_{distance:.4f}")
        print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}")

        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        if abs(distance) < 1e-3:
            if (extend_atom != 0) or extend_xyz != 1:
                print(f"Skip: {name:>40}")
                continue

        if extend_atom >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            continue

        molecular[extend_atom][extend_xyz] += distance
        # data_real = np.load(Path("data/grids_mrks/") / f"data_{name}.npz")

        dft2cc = CC_DFT_DATA(
            molecular,
            name=name,
            basis=args.basis,
            if_basis_str=args.if_basis_str,
        )
        dft2cc.test_mol()
        nocc = dft2cc.mol.nelec[0]

        # 2.1 SCF loop to get the density matrix
        dm1_scf = dft2cc.dm1_dft
        oe_fock = oe.contract_expression(
            "p,p,pa,pb->ab",
            np.shape(dft2cc.ao_0[:, 0]),
            np.shape(dft2cc.ao_0[:, 0]),
            dft2cc.ao_0,
            dft2cc.ao_0,
            constants=[2, 3],
            optimize="optimal",
        )

        def hybrid(new, old, frac_old_=0.8):
            """
            Generate the hybrid density matrix.
            """
            return new * (1 - frac_old_) + old * frac_old_

        for i in range(2500):
            input_mat = dft2cc.grids.vector_to_matrix(
                pyscf.dft.numint.eval_rho(
                    dft2cc.mol,
                    dft2cc.ao_0,
                    dm1_scf,
                )
                + 1e-14
            )
            input_mat = torch.tensor(
                input_mat[:, np.newaxis, :, :], dtype=modeldict.dtype
            ).to("cuda")
            middle_mat = modeldict.model_dict["1"](input_mat).detach().cpu().numpy()
            middle_mat = middle_mat.squeeze(1)
            # middle_mat = data_real["vxc"]

            vxc_scf = dft2cc.grids.matrix_to_vector(middle_mat)
            vxc_mat = oe_fock(vxc_scf, dft2cc.grids.weights, backend="torch")
            vj_scf = np.einsum("qprs,rs->pq", dft2cc.eri, dm1_scf)
            _, mo_scf = np.linalg.eigh(
                dft2cc.mat_hs @ (dft2cc.h1e + vj_scf + vxc_mat) @ dft2cc.mat_hs
            )
            mo_scf = dft2cc.mat_hs @ mo_scf

            dm1_scf_old = dm1_scf.copy()
            dm1_scf = 2 * mo_scf[:, :nocc] @ mo_scf[:, :nocc].T
            error_dm1 = np.linalg.norm(dm1_scf - dm1_scf_old)
            dm1_scf = hybrid(dm1_scf, dm1_scf_old)

            if i % 1 == 0:
                print(
                    f"step:{i:<8}",
                    f"dm: {error_dm1::<10.5e}",
                )
            if (i > 0) and (error_dm1 < 1e-8):
                print(
                    f"step:{i:<8}",
                    f"dm: {error_dm1::<10.5e}",
                )
                break

        # 2.2 check the difference of density (on grids)
        scf_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0,
            dm1_scf,
        )
        cc_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0,
            dft2cc.dm1_cc,
        )
        dft_rho_r = pyscf.dft.numint.eval_rho(
            dft2cc.mol,
            dft2cc.ao_0,
            dft2cc.dm1_dft,
        )
        error_scf_rho_r = np.sum(np.abs(scf_rho_r - cc_rho_r) * dft2cc.grids.weights)
        error_dft_rho_r = np.sum(np.abs(dft_rho_r - cc_rho_r) * dft2cc.grids.weights)
        print(
            f"error_scf_rho_r: {error_scf_rho_r:.2e}, error_dft_rho_r: {error_dft_rho_r:.2e}"
        )

        # 2.3 check the difference of energy (total)
        input_mat = dft2cc.grids.vector_to_matrix(
            pyscf.dft.numint.eval_rho(
                dft2cc.mol,
                dft2cc.ao_0,
                dm1_scf,
            )
            + 1e-14
        )
        input_mat = torch.tensor(
            input_mat[:, np.newaxis, :, :], dtype=modeldict.dtype
        ).to("cuda")
        output_mat = modeldict.model_dict["1"](input_mat).detach().cpu().numpy()
        output_mat = output_mat.squeeze(1)
        # output_mat = data_real["exc_tr_real"]
        
        exc_over_rho_grids = dft2cc.grids.matrix_to_vector(output_mat)
        error_ene_scf = AU2KCALMOL * (
            (
                oe.contract("ij,ji->", dft2cc.h1e, dm1_scf)
                + 0.5 * oe.contract("pqrs,pq,rs->", dft2cc.eri, dm1_scf, dm1_scf)
                + dft2cc.mol.energy_nuc()
                + np.sum(exc_over_rho_grids * scf_rho_r * dft2cc.grids.weights)
            )
            - dft2cc.e_cc
        )
        error_ene_dft = AU2KCALMOL * (dft2cc.e_dft - dft2cc.e_cc)
        print(f"error_scf_ene: {error_ene_scf:.2e}, error_dft_ene: {error_ene_dft:.2e}")
