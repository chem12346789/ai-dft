import numpy as np
import opt_einsum as oe
import pyscf
import scipy.linalg as LA

from cadft.utils.Grids import Grid
from cadft.utils.env_var import DATA_PATH
from cadft.utils_deepks.DataBase import process_input

AU2KJMOL = 2625.5


def deepks(self):
    """
    This idea was copied from the following paper:
    https://arxiv.org/pdf/2012.14615
    """
    # _zeta = 1.5 ** np.array(
    #     [33, 29, 25, 21, 17, 15, 13, 11, 9, 7, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
    # )
    _zeta = 1.5 ** np.array([17, 13, 10, 7, 5, 3, 2, 1, 0, -1, -2, -3])
    _coef = np.diag(np.ones(_zeta.size)) - np.diag(np.ones(_zeta.size - 1), k=1)
    _table = np.concatenate([_zeta.reshape(-1, 1), _coef], axis=1)
    print(_table)
    DEFAULT_BASIS = [
        [0, *_table.tolist()],
        [1, *_table.tolist()],
        [2, *_table.tolist()],
    ]
    fake_mol = pyscf.M(
        atom="O 0 0 0",
        basis=DEFAULT_BASIS,
        verbose=0,
    )
    fake_grids = Grid(fake_mol, level=3)
    fake_ao_value = pyscf.dft.numint.eval_ao(fake_mol, fake_grids.coords)
    print(fake_ao_value.shape)
    fake_mat_s = fake_mol.intor("int1e_ovlp")
    fake_mat_s_inv = LA.inv(fake_mat_s)

    self.mol.verbose = 0
    mf = pyscf.scf.RHF(self.mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.incore_complete = True
    mycc.async_io = False
    mycc.direct = True
    mycc.kernel()

    mdft = pyscf.scf.RKS(self.mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    grids = Grid(self.mol, level=3)
    coords = grids.coords
    weights = grids.weights
    ao_value = pyscf.dft.numint.eval_ao(self.mol, coords, deriv=1)
    dft_dm1 = mdft.make_rdm1()
    dft_r_3 = pyscf.dft.numint.eval_rho(self.mol, ao_value, dft_dm1, xctype="GGA")

    # test the fake molecule
    dft_r = dft_r_3[1]
    c = oe.contract(
        "p, p, pj, jk -> k",
        dft_r,
        fake_grids.weights,
        fake_ao_value,
        fake_mat_s_inv,
    )
    back_rho = oe.contract("k, pk -> p", c, fake_ao_value)
    print(np.sum(np.abs(back_rho * weights - dft_r * weights)))
    print(np.sum((back_rho * weights - dft_r * weights)))

    # data_grids_norm = process_input(dft_r_3, grids)

    # for oxyz in range(1):
    #     for i_atom in range(self.mol.natm):
    #         data = data_grids_norm[oxyz, i_atom, :, :]

    # np.savez_compressed(
    #     DATA_PATH / f"data_{self.name}.npz",
    #     weights=grids.vector_to_matrix(weights),
    #     eigenval=eigenval,
    #     eigenvector=eigenvector,
    #     rho_inv_4_norm=data_grids_norm,
    #     delta_ene=mycc.e_tot - mdft.e_tot,
    # )
