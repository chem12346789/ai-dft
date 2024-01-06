"""@package docstring
Documentation for this module.
 
More details.
"""

import sys
import numpy as np
import torch
import opt_einsum as oe


def gen_w_vec(dm1, dm1_r, ao_0, ao_1, vxc, coords):
    """
    Documentation for a function.

    The function is used to generate the w vector. The w vector is used to check the convergence of the MRKS equations.
    """
    part = oe.contract(
        "pm,mn,kpn->pk",
        ao_0,
        dm1,
        ao_1,
    )
    r_dot_dev_dm1_r = np.zeros_like(dm1_r)
    r_dot_dev_dm1_r += part[:, 0] * coords[:, 0]
    r_dot_dev_dm1_r += part[:, 1] * coords[:, 1]
    r_dot_dev_dm1_r += part[:, 2] * coords[:, 2]
    return (3 * dm1_r + 2 * r_dot_dev_dm1_r) * vxc
