""""""
import argparse
import logging
import torch
import numpy as np
import opt_einsum as oe
from scipy import linalg as LA
from pathlib import Path

import pyscf
from pyscf import dft

from src.mrks_pyscf.utils.grids import Grid
from src.aidft.unet.unet_model import UNet
from src.aidft.get_args import get_args_quantum, get_args_model
from src.mrks_pyscf.utils.grids import rotate


def predict_potential(net, input_data, device):
    """Documentation for a function.

    More details.
    """
    net.eval()
    input_data = torch.from_numpy(input_data)
    input_data = input_data.unsqueeze(0)
    input_data = input_data.to(device=device, dtype=torch.float64)

    with torch.no_grad():
        output = net(input_data).cpu()

    return output.numpy()


def old_function(distance):
    if distance < 1.5:
        return 0.8
    if distance < 2.5:
        return 0.95
    if distance < 3.5:
        return 0.975


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--distance_list",
        "-dl",
        nargs="+",
        type=float,
        help="Distance between atom H to the origin. Default is 1.0.",
        default=1.0,
    )
    get_args_quantum(parser)
    get_args_model(parser)
    args = parser.parse_args()

    if len(args.distance_list) == 3:
        distance_l = np.linspace(
            args.distance_list[0], args.distance_list[1], int(args.distance_list[2])
        )
    else:
        distance_l = args.distance

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    net.double()

    device = torch.device("cuda")
    model_path = Path(args.load) / "checkpoints" / "checkpoint_epoch.pth"
    logging.info("Loading model %s", f"{model_path}")
    logging.info("Using device %s", f"{device}")

    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    logging.info("Model loaded!")
    net.to(device=device)

    with open(
        f"./output-{args.qm_method}-{args.basis_set}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for distance in distance_l:
            FRAC_OLD = old_function(distance)

            mol = pyscf.M(
                atom=[["H", distance, 0, 0], ["H", 0, 0, 0]],
                basis=args.basis_set,
            )

            mol = rotate(mol, angle_list=args.rotate)
            print(mol.atom)

            myhf = pyscf.scf.HF(mol)
            myhf.kernel()
            h1e = myhf.get_hcore()
            mo = myhf.mo_coeff

            mdft = mol.KS()
            mdft.xc = "b3lyp"
            mdft.kernel()
            dm1_compare = mdft.make_rdm1()
            e_compare = mdft.e_tot

            mycisd = pyscf.ci.CISD(myhf).run()
            dm1_cisd = mycisd.make_rdm1()
            dm1_cisd = oe.contract("ij,pi,qj->pq", dm1_cisd, mo, mo)
            e_cisd = mycisd.e_tot

            s_0_ao = mol.intor("int1e_ovlp")
            inv_half_ovlp = LA.fractional_matrix_power(s_0_ao, -0.5).real
            nocc = mol.nelec[0]

            grids = Grid(mol, args.level)
            dm1 = dm1_compare.copy()
            ni = dft.numint.NumInt()
            ao_0 = ni.eval_ao(mol, grids.coords, deriv=0)

            FLAG = True
            STEP = 0
            while FLAG:
                STEP += 1

                rho_0_grid = grids.vector_to_matrix(
                    oe.contract(
                        "uv,gu,gv->g",
                        dm1,
                        ao_0,
                        ao_0,
                        optimize="optimal",
                    )
                )

                vxc_wy_grid = np.zeros_like(rho_0_grid)
        
                # # trick here, use data from the training set.
                # vxc_wy_grid[0, :, :] = np.load(
                #     Path(args.load)
                #     / "data"
                #     / "masks"
                #     / "data-HH-cc-pcvqz-3-0.5000-0.npy"
                # )
                # vxc_wy_grid[1, :, :] = np.load(
                #     Path(args.load)
                #     / "data"
                #     / "masks"
                #     / "data-HH-cc-pcvqz-3-0.5000-1.npy"
                # )

                for i_atom, rho_0_grid_atom in enumerate(rho_0_grid):
                    vxc_wy_grid[i_atom] = predict_potential(
                        net,
                        rho_0_grid_atom.reshape(
                            1, rho_0_grid.shape[1], rho_0_grid.shape[2]
                        ),
                        device,
                    )[0, 0, :, :]

                vxc = grids.matrix_to_vector(vxc_wy_grid)
                xc_v = oe.contract(
                    "p,p,pa,pb->ab",
                    vxc,
                    grids.weights,
                    ao_0,
                    ao_0,
                    optimize="optimal",
                )
                vj = pyscf.scf.hf.get_jk(mol, dm1)[0]
                fock_a = inv_half_ovlp @ (h1e + vj + xc_v) @ inv_half_ovlp
                eigvecs, mo = np.linalg.eigh(fock_a)
                mo = inv_half_ovlp @ mo
                dm1_old = dm1.copy()
                dm1 = 2 * mo[:, :nocc] @ mo[:, :nocc].T
                error = np.linalg.norm(dm1 - dm1_old)
                print(f"error of dm1, {error:.2e}")
                if (error < 1e-8) or (STEP > 2500):
                    FLAG = False
                dm1 = dm1 * (1 - FRAC_OLD) + dm1_old * FRAC_OLD

            dm1_compare_r = oe.contract(
                "uv,gu,gv->g",
                dm1_compare,
                ao_0,
                ao_0,
                optimize="optimal",
            )

            dm1_cisd_r = oe.contract(
                "uv,gu,gv->g",
                dm1_cisd,
                ao_0,
                ao_0,
                optimize="optimal",
            )

            dm1_r = oe.contract(
                "uv,gu,gv->g",
                dm1,
                ao_0,
                ao_0,
                optimize="optimal",
            )

            error_compare = np.sum(abs(dm1_compare_r - dm1_cisd_r) * grids.weights)
            error_scf = np.sum(abs(dm1_r - dm1_cisd_r) * grids.weights)

            str_ = (
                f"error of error_compare, {error_compare:<10.2e}"
                f"error of error_scf, {error_scf:<10.2e}"
            )

            # rho_0_grid = grids.vector_to_matrix(
            #     oe.contract(
            #         "uv,gu,gv->g",
            #         dm1,
            #         ao_0,
            #         ao_0,
            #         optimize="optimal",
            #     )
            # )

            # rho_0_grid[0, :, :] = np.load(
            #     Path(args.load) / "data" / "imgs" / "0.50-0.npy"
            # )[0, :, :]
            # rho_0_grid[1, :, :] = np.load(
            #     Path(args.load) / "data" / "imgs" / "0.50-1.npy"
            # )[0, :, :]
            # # rho_0_grid[0, :, :] += 1e-3

            # e_grid = np.zeros_like(rho_0_grid)
            # for i_atom, rho_0_grid_atom in enumerate(rho_0_grid):
            #     e_grid[i_atom] = predict_potential(
            #         net,
            #         rho_0_grid_atom.reshape(
            #             1, rho_0_grid.shape[1], rho_0_grid.shape[2]
            #         ),
            #         device,
            #     )[0, 0, :, :]
            # e_nuc = oe.contract("ij,ji->", h1e, dm1).real
            # e_vj = oe.contract("ij,ji->", myhf.get_jk(mol, dm1, 1)[0], dm1).real

            # exc_kin_correct = grids.matrix_to_vector(e_grid)

            # e_grid_check = np.zeros_like(e_grid)
            # e_grid_check[0, :, :] = np.load(
            #     Path(args.load) / "data" / "masks" / "0.50-0.npy"
            # )[0, :, :]
            # e_grid_check[1, :, :] = np.load(
            #     Path(args.load) / "data" / "masks" / "0.50-1.npy"
            # )[0, :, :]
            # exc_kin_correct_check = grids.matrix_to_vector(e_grid_check)

            # print(e_grid[0, -1, :])
            # print(e_grid_check[0, -1, :])
            # print(np.shape(e_grid_check))

            # print((exc_kin_correct * dm1_r).sum())
            # print((exc_kin_correct_check * dm1_r).sum())

            # ene_t_vc = mol.energy_nuc() + e_nuc + e_vj * 0.5 + (exc_kin_correct).sum()

            # AU_TO_KJMOL = 1
            # str_ = (
            #     f"e_nuc: {e_nuc}\n"
            #     f"e_vj: {e_vj}\n"
            #     f"nuc_rep: {mol.energy_nuc()}\n"
            #     f"exc: {exc_kin_correct.sum()}\n"
            #     f"error of energy: {((ene_t_vc - e_cisd) * AU_TO_KJMOL):<10.2e}\n"
            #     f"error of b3lyp energy:, {((e_compare - e_cisd) * AU_TO_KJMOL):<10.2e}\n"
            # )

            print(str_)
            # print(str__)
            f.writelines(f"{distance:.3f}\t" f"{str_}\n")
            f.flush()
