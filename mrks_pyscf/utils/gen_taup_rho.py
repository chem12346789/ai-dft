"""@package docstring
Documentation for this module.
 
More details.
"""

import sys
import numpy as np
import torch
import opt_einsum as oe


def gen_taup_rho(
    oe_taup_rho, dm1_r, eigs_v_dm1, eigs_e_dm1, backend="numpy", logger=None
):
    """Documentation for a function.

    More details.
    """
    taup = np.zeros(len(dm1_r))
    norb = np.shape(eigs_v_dm1)[1]

    for i in range(norb):
        for j in range(i + 1):
            if logger is not None:
                setp = i * (i + 1) // 2 + j
                if setp % 100 == 0:
                    logger.info(f"\nstep:{setp:<8} of {norb * (norb + 1) // 2:<8}")
                if setp % 10 == 0:
                    logger.info(".")
            if i != j:
                part = oe_taup_rho(eigs_v_dm1[:, i], eigs_v_dm1[:, j], backend=backend)
                part -= oe_taup_rho(eigs_v_dm1[:, j], eigs_v_dm1[:, i], backend=backend)
                part1 = torch.sum(part**2, axis=1).cpu().numpy()
                taup += part1 * eigs_e_dm1[i] * eigs_e_dm1[j]
    taup_rho = taup / dm1_r**2 * 0.5
    return taup_rho


def gen_tau_rho(
    oe_tau_rho, dm1_r, eigs_v_dm1, eigs_e_dm1, backend="numpy", logger=None
):
    """Documentation for a function.

    More details.
    """
    taup = np.zeros(len(dm1_r))
    norb = np.shape(eigs_v_dm1)[1]

    for i in range(norb):
        if logger is not None:
            setp = i
            if setp % 100 == 0:
                logger.info(f"\nstep:{setp:<8} of {norb:<8}")
            if setp % 10 == 0:
                logger.info(".")
        part = oe_tau_rho(eigs_v_dm1[:, i], eigs_v_dm1[:, i], backend=backend) * 2
        taup += part.cpu().numpy() * eigs_e_dm1[i]
    taup_rho = -taup * 0.5
    return taup_rho
