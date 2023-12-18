"""@package docstring
Documentation for this module.
 
More details.
"""

import pyscf
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import ccsd_t_slow as ccsd_t


def kernel(method, myhf):
    """Documentation for a function.

    More details.
    """
    if method == "fci":
        cisolver = pyscf.fci.FCI(myhf)
        e, fcivec = cisolver.kernel()
        dm1_mo, dm2_mo = cisolver.make_rdm12(
            fcivec, myhf.mo_coeff.shape[1], myhf.mol.nelectron
        )
    elif method == "ccsd":
        mycc = pyscf.cc.CCSD(myhf).run()
        e = mycc.e_tot
        dm1_mo = mycc.make_rdm1()
        dm2_mo = mycc.make_rdm2()
    elif method == "ccsdt":
        mycc = pyscf.cc.CCSD(myhf)
        mycc.conv_tol = 1e-12
        _, t1, t2 = mycc.kernel()
        eris = mycc.ao2mo()

        e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
        e = mycc.e_tot + e3ref
        l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
        dm1_mo = ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris)
        dm2_mo = ccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris=eris)
    elif method == "cisd":
        myci = pyscf.ci.CISD(myhf).run()
        dm1_mo = myci.make_rdm1()
        dm2_mo = myci.make_rdm2()
        e = myci.e_tot
    else:
        raise NotImplementedError

    return e, dm1_mo, dm2_mo
