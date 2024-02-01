""""""
import argparse
from pathlib import Path
import json
import torch
import numpy as np
import opt_einsum as oe

import pyscf

from mrks_pyscf.mrksinv import Mrksinv
from mrks_pyscf.utils.mol import old_function
from mrks_pyscf.utils.logger import gen_logger
from mrks_pyscf.utils.mol import Mol
from aidft import parser_model


def predict_potential(net, input_data, device):
    """
    Documentation for a function.

    More details.
    """
    net.eval()
    input_data = torch.from_numpy(input_data)
    input_data = input_data.unsqueeze(0)
    input_data = input_data.to(device=device, dtype=torch.float64)

    with torch.no_grad():
        output = net(input_data).cpu()

    return output.numpy()


path = Path(__file__).resolve().parents[1] / "data"
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
parser_model(parser)
args = parser.parse_args()

distance_l, logger, path_dir = gen_logger(
    args.distance_list,
    f"{args.molecular}-{args.basis}-{args.level}-predict",
    path,
)
molecular = Mol[args.molecular]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "unet_small":
    from aidft import UNet_small as UNet
elif args.model == "unet":
    from aidft import UNet
else:
    raise ValueError("Unknown model")

net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
net.double()
net = net.to(memory_format=torch.channels_last)

if args.load:
    dir_checkpoint = Path(args.name) / "checkpoints/"
    load_path = (
        dir_checkpoint
        / f"checkpoint_epoch-{args.optimizer}-{args.scheduler}-{args.load}.pth"
    )
    state_dict = torch.load(load_path, map_location=device)
    net.load_state_dict(state_dict)
    logger.info("Model loaded from %s\n", load_path)

net.to(device=device)

save_data = {}

for distance in distance_l:
    molecular[0][1] = distance
    save_data[distance] = {}
    logger.info("%s", f"The distance is {distance}.")
    FRAC_OLD = old_function(distance, args.old_factor_scheme, args.old_factor)

    mrks_inv = Mrksinv(
        molecular,
        path=path_dir / f"{distance:.4f}",
        args=None,
        logger=logger,
        frac_old=FRAC_OLD,
        level=args.level,
        basis=args.basis,
    )

    myhf = pyscf.scf.HF(mrks_inv.mol)
    myhf.kernel()
    h1e = myhf.get_hcore()
    mo = myhf.mo_coeff

    mdft = mrks_inv.mol.KS()
    mdft.xc = "b3lyp"
    mdft.kernel()
    dm1_compare = mdft.make_rdm1()
    e_compare = mdft.e_tot

    mycisd = pyscf.ci.CISD(myhf).run()
    dm1_cisd = mycisd.make_rdm1()
    dm1_cisd = oe.contract("ij,pi,qj->pq", dm1_cisd, mo, mo)
    e_cisd = mycisd.e_tot

    dm1 = dm1_cisd.copy()
    vj = mrks_inv.myhf.get_jk(mrks_inv.mol, dm1, 1)[0]

    for step in range(args.scf_step):
        dm1_r = mrks_inv.aux_function.oe_rho_r(dm1)
        dm1_r_grid = mrks_inv.grids.vector_to_matrix(dm1_r)
        mrks_inv.logger.info(
            f"dm1_r_grid, {np.array2string(dm1_r_grid, precision=4, separator=',', suppress_small=True)}\n"
        )

        # # trick here, use data from the training set.
        # dm1_r_grid[0, :, :] = np.load(
        #     Path(args.name)
        #     / "data"
        #     / "imgs"
        #     / f"data-HH-cc-pcvqz-cisd-4-{distance:.4f}-0.npy"
        # )
        # dm1_r_grid[1, :, :] = np.load(
        #     Path(args.name)
        #     / "data"
        #     / "imgs"
        #     / f"data-HH-cc-pcvqz-cisd-4-{distance:.4f}-1.npy"
        # )
        # mrks_inv.logger.info(
        #     "dm1_r_grid, %s\n",
        #     f"{np.array2string(dm1_r_grid, precision=4, separator=',', suppress_small=True)}",
        # )

        vxc_grid = np.zeros_like(dm1_r_grid)
        vxc_grid[0, :, :] = np.load(
            Path(args.name)
            / "data"
            / "masks"
            / f"data-HH-cc-pcvqz-cisd-4-{distance:.4f}-0.npy"
        )
        vxc_grid[1, :, :] = np.load(
            Path(args.name)
            / "data"
            / "masks"
            / f"data-HH-cc-pcvqz-cisd-4-{distance:.4f}-1.npy"
        )

        mrks_inv.logger.info(
            f"vxc_grid, {np.array2string(vxc_grid, precision=4, separator=',', suppress_small=True)}\n"
        )

        for i_atom, rho_0_grid_atom in enumerate(dm1_r_grid):
            vxc_grid[i_atom] = predict_potential(
                net,
                rho_0_grid_atom.reshape(1, dm1_r_grid.shape[1], dm1_r_grid.shape[2]),
                device,
            )[0, 0, :, :]

        mrks_inv.logger.info(
            f"vxc_grid, {np.array2string(vxc_grid, precision=4, separator=',', suppress_small=True)}"
        )

        vxc = mrks_inv.grids.matrix_to_vector(vxc_grid)
        xc_v = mrks_inv.aux_function.oe_fock(vxc, mrks_inv.grids.weights)
        vj = mrks_inv.hybrid(mrks_inv.myhf.get_jk(mrks_inv.mol, dm1, 1)[0], vj)
        fock_a = mrks_inv.mat_hs @ (h1e + vj + xc_v) @ mrks_inv.mat_hs
        eigvecs, mo = np.linalg.eigh(fock_a)
        mo = mrks_inv.mat_hs @ mo
        dm1_old = dm1.copy()
        dm1 = 2 * mo[:, : mrks_inv.nocc] @ mo[:, : mrks_inv.nocc].T
        error = np.linalg.norm(dm1 - dm1_old)
        if args.noisy_print:
            mrks_inv.logger.info(f"\nerror of dm1, {error:.4e}\n")
        else:
            if step % 100 == 0:
                mrks_inv.logger.info(f"\nerror of dm1, {error:.4e}\n")
            elif step % 10 == 0:
                mrks_inv.logger.info(".")

        if error < args.error_scf:
            break

    dm1_compare_r = mrks_inv.aux_function.oe_rho_r(dm1_compare)
    dm1_cisd_r = mrks_inv.aux_function.oe_rho_r(dm1_cisd)

    error_compare = np.sum(abs(dm1_compare_r - dm1_cisd_r) * mrks_inv.grids.weights)
    error_scf = np.sum(abs(dm1_r - dm1_cisd_r) * mrks_inv.grids.weights)

    save_data[distance]["energy_inv"] = error_compare
    save_data[distance]["energy_inv"] = error_scf

    mrks_inv.logger.info(f"\nCheck! {error_compare:16.10f}\n" f"{error_scf:21.10f}\n")

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

with open(mrks_inv.path / "save_data.json", "w", encoding="utf-8") as f:
    json.dump(save_data, f, indent=4)
