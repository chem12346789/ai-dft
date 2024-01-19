import logging
from pathlib import Path
import numpy as np
import pyscf
import argparse

from mrks_pyscf.mrksinv import Mrksinv
from mrks_pyscf.utils.mol import Mol, BASIS

path = Path(__file__).resolve().parents[1] / "data"
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)

parser.add_argument(
    "--molecular",
    "-m",
    type=str,
    default="HH",
    help="Name of molecular.",
)

parser.add_argument(
    "--basis",
    "-b",
    type=str,
    default="cc-pv5z",
    help="Name of basis. We use cc-pv5z as default. Note we will remove core correlation of H atom; See https://github.com/pyscf/pyscf/issues/1795",
)

parser.add_argument(
    "--level",
    "-l",
    type=int,
    help="Level of DFT grid. Default is 4.",
    default=4,
)

parser.add_argument(
    "--distance_list",
    "-dl",
    nargs="+",
    type=float,
    help="Distance between atom H to the origin. Default is 1.0.",
    default=1.0,
)

parser.add_argument(
    "--old_factor_scheme",
    "-fs",
    type=int,
    help="Scheme for old factor. Default is 1. -1 means use given old factor.",
    default=-1,
    choices=[-1, 1, 2],
)

parser.add_argument(
    "--old_factor",
    "-f",
    type=float,
    help="Old factor. Default is 0.9.",
    default=0.9,
)

parser.add_argument(
    "--device",
    "-de",
    type=str,
    choices=["cpu", "cuda"],
    help="Device for inversion. Default is 'cuda'.",
    default="cuda",
)

parser.add_argument(
    "--method",
    "-me",
    type=str,
    choices=["cisd", "fci", "ccsd", "ccsdt"],
    help="Method for quantum chemistry calculation. Default is 'cisd'.",
    default="cisd",
)

args = parser.parse_args()

if args.old_factor_scheme == 1:
    from src.mrks_pyscf.utils.mol import old_function1 as old_function
elif args.old_factor_scheme == 2:
    from src.mrks_pyscf.utils.mol import old_function2 as old_function
else:

    def old_function(distance):
        """
        This function is used to determine the factor of mixing old and new density matrix in SCF process
        """
        return old_factor


if len(args.distance_list) == 3:
    distance_l = np.linspace(
        args.distance_list[0], args.distance_list[1], int(args.distance_list[2])
    )
else:
    distance_l = args.distance

molecular = Mol[args.molecular]

path_dir = path / f"data-{args.molecular}-{args.basis}-{args.method}-{args.level}"
if not path_dir.exists():
    path_dir.mkdir(parents=True)

logger = logging.getLogger(__name__)
logging.StreamHandler.terminator = ""
# clear the log
Path(path_dir / "compare.log").unlink(missing_ok=True)
logger.addHandler(logging.FileHandler(path_dir / "compare.log"))
logger.setLevel(logging.DEBUG)

for distance in distance_l:
    molecular[0][1] = distance
    logger.info("%s", f"The distance is {distance}.")

    basis = {}

    for i_atom in molecular:
        basis[i_atom[0]] = (
            BASIS[args.basis]
            if ((i_atom[0] == "H") and (args.basis in BASIS))
            else args.basis
        )

    mol = pyscf.M(
        atom=molecular,
        basis=basis,
    )

    mrks_inv = Mrksinv(
        mol,
        frac_old=old_function(distance),
        level=args.level,
        scf_step=25000,
        logger=logger,
        device=args.device,
    )
    mrks_inv.kernel(method=args.method, gen_dm2=False)
    print("Inverse done")

    mdft = mol.KS()
    mdft.xc = "b3lyp"
    mdft.kernel()
    dm1_dft = mdft.make_rdm1()

    mrks_inv.vxc = mrks_inv.grids.matrix_to_vector(
        np.load(path_dir / f"{distance:.4f}" / "mrks.npy")
    )
    dm1_inv = 2 * np.load(path_dir / f"{distance:.4f}" / "dm1_inv.npy")
    dm1_scf = mrks_inv.scf(dm1_dft)

    dm1_exa_r = mrks_inv.aux_function.oe_rho_r(mrks_inv.dm1, backend="torch")
    dm1_dft_r = mrks_inv.aux_function.oe_rho_r(dm1_dft, backend="torch")
    dm1_inv_r = mrks_inv.aux_function.oe_rho_r(dm1_inv, backend="torch")
    dm1_scf_r = mrks_inv.aux_function.oe_rho_r(dm1_scf, backend="torch")
    dm1_dft_error = np.sum(np.abs(dm1_exa_r - dm1_dft_r) * mrks_inv.grids.weights)
    dm1_inv_error = np.sum(np.abs(dm1_exa_r - dm1_inv_r) * mrks_inv.grids.weights)
    dm1_scf_error = np.sum(np.abs(dm1_exa_r - dm1_scf_r) * mrks_inv.grids.weights)
    dm1_scf_inv_error = np.sum(np.abs(dm1_inv_r - dm1_scf_r) * mrks_inv.grids.weights)
    logger.info("%s", f"dm1_dft_error: {dm1_dft_error:16.10e}\n")
    logger.info("%s", f"dm1_inv_error: {dm1_inv_error:16.10e}\n")
    logger.info("%s", f"dm1_scf_error: {dm1_scf_error:16.10e}\n")
    logger.info("%s", f"dm1_scf_inv_error: {dm1_scf_inv_error:16.10e}\n")

    exc_kin_over_dm = mrks_inv.grids.matrix_to_vector(
        np.load(path_dir / f"{distance:.4f}" / "mrks_e_dm.npy")
    )
    logger.info(
        "%s",
        f"energy_inv: {2625.5 * mrks_inv.gen_energy(dm1_inv, exc_kin_over_dm):16.10f}\n",
    )
    logger.info(
        "%s",
        f"energy_scf: {2625.5 * mrks_inv.gen_energy(dm1_scf, exc_kin_over_dm):16.10f}\n",
    )
    logger.info(
        "%s",
        f"energy_exa: {2625.5 * mrks_inv.gen_energy(mrks_inv.dm1, exc_kin_over_dm):16.10f}\n",
    )
    logger.info("%s", f"ene_dft: {2625.5 * mdft.e_tot:16.10f}\n")
    logger.info("%s", f"ene_exa: {2625.5 * mrks_inv.e:16.10f}\n")
    logger.info("\n")
    del mrks_inv
    del mdft
