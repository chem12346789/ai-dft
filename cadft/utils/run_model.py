"""
This module is used to run the model and get the results.
"""

import numpy as np
import torch

from cadft.utils.nao import NAO


def gen_dm1(dft2cc, dm1, model_dict, device):
    dm1_cc = np.zeros((dft2cc.mol.nao, dft2cc.mol.nao))
    for i in range(dft2cc.mol.natm):
        for j in range(dft2cc.mol.natm):
            atom_name = dft2cc.atom_info["atom"][i] + dft2cc.atom_info["atom"][j]
            input_mat = (
                dm1[dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]]
            ).flatten()
            input_mat = (
                torch.as_tensor(input_mat.copy())
                .to(torch.float64)
                .contiguous()
                .to(device=device)
                .requires_grad_(True)
            )
            output_mat = model_dict[atom_name + "1"](input_mat)
            dm1_cc[dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]] = (
                output_mat.detach()
                .cpu()
                .numpy()
                .reshape(
                    NAO[dft2cc.atom_info["atom"][i]], NAO[dft2cc.atom_info["atom"][j]]
                )
            )

    return dm1_cc


def gen_f_mat(dft2cc, dm1_cc, model_dict, device):
    f_mat = np.zeros((dft2cc.mol.nao, dft2cc.mol.nao))
    ene_xc = 0
    for i in range(dft2cc.mol.natm):
        for j in range(dft2cc.mol.natm):
            atom_name = dft2cc.atom_info["atom"][i] + dft2cc.atom_info["atom"][j]
            input_mat = (
                dm1_cc[dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]]
            ).flatten()
            input_mat = (
                torch.as_tensor(input_mat.copy())
                .to(torch.float64)
                .contiguous()
                .to(device=device)
                .requires_grad_(True)
            )
            e_xc = model_dict[atom_name + "2"](input_mat)
            grad_dms = torch.autograd.grad(e_xc, input_mat)
            f_mat[dft2cc.atom_info["slice"][i], dft2cc.atom_info["slice"][j]] = (
                grad_dms[0]
                .detach()
                .cpu()
                .numpy()
                .reshape(
                    NAO[dft2cc.atom_info["atom"][i]], NAO[dft2cc.atom_info["atom"][j]]
                )
            )
            ene_xc += e_xc.detach().cpu().numpy()[0]

    return f_mat, ene_xc
