import numpy as np
import pyscf

from cadft.utils.Grids import Grid
from cadft.utils.env_var import DATA_PATH

AU2KJMOL = 2625.5


def umrks_append(self):
    """
    Append the data to the existing npz file.
    """
    for i_spin in range(2):
        if not (DATA_PATH / f"data_{self.name}_{i_spin}.npz").exists():
            print(
                f"File {DATA_PATH / f'data_{self.name}_{i_spin}.npz'} does not exist."
            )
            return

        data = np.load(DATA_PATH / f"data_{self.name}_{i_spin}.npz")

        grids = Grid(self.mol)
        ao_value = pyscf.dft.numint.eval_ao(self.mol, grids.coords)
        inv_r = pyscf.dft.numint.eval_rho(self.mol, ao_value, data["dm_inv"])
        evxc_lda = pyscf.dft.libxc.eval_xc("lda,vwn", inv_r)

        np.savez_compressed(
            DATA_PATH / f"data_{self.name}_{i_spin}.npz",
            dm_cc=data["dm_cc"],
            dm_inv=data["dm_inv"],
            rho_inv=data["rho_inv"],
            weights=data["weights"],
            vxc=data["vxc"],
            exc=data["exc"],
            exc_real=data["exc_real"],
            exc1_tr=data["exc1_tr"],
            rho_inv_4_norm=data["rho_inv_4_norm"],
            exc1_tr_lda=data["exc1_tr"] - grids.vector_to_matrix(evxc_lda[0]),
            vxc1_lda=data["vxc"] - grids.vector_to_matrix(evxc_lda[1][0]),
        )
