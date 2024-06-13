"""@package docstring
Documentation for this module.
 
More details.
"""

import numpy as np
import torch


def gen_taup_rho(
    dm1_r,
    eigs_v_dm1,
    eigs_e_dm1,
    oe_taup_rho,
    backend="numpy",
):
    """
    Documentation for a function.

    More details.
    """
    taup = np.zeros(len(dm1_r))
    norb = np.shape(eigs_v_dm1)[1]

    for i in range(norb):
        for j in range(i + 1):
            if i != j:
                part = oe_taup_rho(eigs_v_dm1[:, i], eigs_v_dm1[:, j], backend=backend)
                part -= oe_taup_rho(eigs_v_dm1[:, j], eigs_v_dm1[:, i], backend=backend)
                part1 = np.sum(part**2, axis=1)
                taup += part1 * eigs_e_dm1[i] * eigs_e_dm1[j]
    taup_rho = taup / (dm1_r + 1e-14) * 0.5
    return taup_rho


def gen_tau_rho(
    dm1_r,
    eigs_v_dm1,
    eigs_e_dm1,
    oe_tau_rho,
    backend="numpy",
):
    """
    Documentation for a function.

    More details.
    """
    tau = np.zeros(len(dm1_r))
    norb = np.shape(eigs_v_dm1)[1]

    for i in range(norb):
        part = oe_tau_rho(eigs_v_dm1[:, i], eigs_v_dm1[:, i], backend=backend)
        tau += part * eigs_e_dm1[i]
    taup_rho = -tau / 2
    return taup_rho
