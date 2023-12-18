"""@package docstring
Documentation for this module.
 
More details.
"""

import opt_einsum as oe
import numpy as np


class Auxfunction:
    """
    This class is modified from pyscf.dft.gen_grid.Grids. Some default parameters are changed.
    """

    def __init__(self, mrks_inv):
        self.oe_rho_r = oe.contract_expression(
            "uv,gu,gv->g",
            (mrks_inv.norb, mrks_inv.norb),
            mrks_inv.ao_0,
            mrks_inv.ao_0,
            constants=[1, 2],
            optimize="auto",
        )

        self.oe_fock = oe.contract_expression(
            "p,p,pa,pb->ab",
            np.shape(mrks_inv.ao_0[:, 0]),
            np.shape(mrks_inv.ao_0[:, 0]),
            mrks_inv.ao_0,
            mrks_inv.ao_0,
            constants=[2, 3],
            optimize="auto",
        )

        self.oe_taup_rho = oe.contract_expression(
            "pm,m,n,kpn->pk",
            mrks_inv.ao_0,
            (mrks_inv.norb,),
            (mrks_inv.norb,),
            mrks_inv.ao_1,
            constants=[0, 3],
            optimize="auto",
        )

        self.oe_ebar_r_wf = oe.contract_expression(
            "i,mi,ni,pm,pn->p",
            (mrks_inv.norb,),
            (mrks_inv.norb, mrks_inv.norb),
            (mrks_inv.norb, mrks_inv.norb),
            mrks_inv.ao_0,
            mrks_inv.ao_0,
            constants=[3, 4],
            optimize="auto",
        )

        self.oe_ebar_r_ks = oe.contract_expression(
            "i,mi,ni,pm,pn->p",
            (mrks_inv.nocc,),
            (mrks_inv.norb, mrks_inv.nocc),
            (mrks_inv.norb, mrks_inv.nocc),
            mrks_inv.ao_0,
            mrks_inv.ao_0,
            constants=[3, 4],
            optimize="auto",
        )
