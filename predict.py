""""""

import argparse
from pathlib import Path
import json
import gc
from datetime import datetime
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import opt_einsum as oe

import pyscf
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import ccsd_t_slow as ccsd_t

from mrks_pyscf.mrksinv import Mrksinv
from mrks_pyscf.utils.mol import old_function
from mrks_pyscf.utils.logger import gen_logger
from mrks_pyscf.utils.mol import Mol, PREDICT_MOLECULAR
from mrks_pyscf.utils.grids import rotate
from aidft import numpy2str, parser_model, gen_model


def predict_potential(
    net,
    input_data,
    device,
    if_pad=True,
    if_flatten=True,
    if_sort=False,
):
    """
    Documentation for a function.

    More details.
    """
    if if_sort:
        data_sort = np.zeros_like(input_data, dtype=float)
        data_sort_index = np.zeros_like(input_data, dtype=int)
        data_sort_back_index = np.zeros_like(input_data, dtype=int)
        for i in range(input_data.shape[0]):
            for j in range(input_data.shape[1]):
                data_sort_index[i, j, :] = np.argsort(
                    np.sum(input_data[i, :, :], axis=0)
                )
                for k in range(input_data.shape[2]):
                    data_sort_back_index[i, j, data_sort_index[i, j, k]] = k
                data_sort[i, j, :] = input_data[i, j, data_sort_index[i, j, :]]
        input_data = data_sort.copy()

    net.eval()
    input_data = torch.from_numpy(input_data)
    input_data = input_data.unsqueeze(0)

    if if_pad:
        input_data = F.pad(input_data, (9, 9, 10, 11), "reflect")
    if if_flatten:
        input_data = np.squeeze(input_data, axis=0)

    input_data = input_data.to(device=device, dtype=torch.float64)

    with torch.no_grad():
        output = net(input_data).cpu().numpy()

    if if_flatten:
        output = output.reshape(1, 2, 75, 302)
    else:
        output = output[:, :, 10:-11, 9:-9]

    if if_sort:
        data_orig = np.zeros_like(output)
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                data_orig[0, i, j, :] = output[0, i, j, data_sort_back_index[0, j, :]]
        output = data_orig.copy()
    return output


path = Path(__file__).resolve().parents[1] / "data"
parser = argparse.ArgumentParser(
    description="Predict the potential with the trained model"
)
parser_model(parser)
args = parser.parse_args()

distance_l, logger, path_dir = gen_logger(
    args.distance_list,
    f"{args.molecular}-predict",
    path,
)
molecular = Mol[args.molecular]
# logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "weit" in args.name:
    WEIGHT = True
else:
    WEIGHT = False
if "small" in args.name:
    SMALL = True
else:
    SMALL = False
if "1" in args.name:
    SMP = True
else:
    SMP = False

model_dict = {}
# XC_CODE = "SVWN"
# XC_CODE = "lda,"
XC_CODE = None

if args.load:
    for atom in args.ai_list:
        if atom not in args.molecular:
            continue

        model = gen_model(args)
        model.double()
        model = model.to(memory_format=torch.channels_last)
        if args.molecular in PREDICT_MOLECULAR:
            dir_checkpoint = (
                Path(f"mrks-e-{PREDICT_MOLECULAR[args.molecular]}-{atom}-{args.name}")
                / "checkpoints/"
            )
        else:
            dir_checkpoint = (
                Path(f"mrks-e-{args.molecular}-{atom}-{args.name}") / "checkpoints/"
            )

        if args.load == 1:
            list_of_path = dir_checkpoint.glob("*.pth")
            load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
        else:
            load_path = dir_checkpoint / f"rmsprop-plateau-{args.load}.pth"

        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model loaded from %s\n", load_path)
        #  show time of the checkpoint
        logger.info("Time: %s\n", datetime.fromtimestamp(load_path.stat().st_mtime))
        copy_file = (
            path_dir
            / "saved_model"
            / f"{args.molecular}-{atom}-{datetime.now():%b_%d_%H_%M_%S_%Y}.pth"
        )
        if not (path_dir / "saved_model").exists():
            (path_dir / "saved_model").mkdir(parents=True)
        shutil.copy(load_path, copy_file)

        model.to(device=device)
        model_dict[atom] = model

for item in Path(path_dir).glob("*/save_data.json"):
    if item.is_file():
        item.unlink()

for distance in distance_l:
    molecular[0][1] = distance
    if "sort" in args.name:
        rotate(molecular, "random")
    save_data = {}
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
        if_basis_str=True,
    )

    myhf = pyscf.scf.HF(mrks_inv.mol)
    myhf.kernel()
    h1e = myhf.get_hcore()

    mdft = mrks_inv.mol.KS()
    mdft.xc = "b3lyp"
    mdft.kernel()
    dm1_dft = mdft.make_rdm1()
    e_dft = mdft.e_tot
    dm1 = dm1_dft.copy()

    for iatom, atom_list in enumerate(mrks_inv.mol.atom):
        atom = atom_list[0]
        if atom in args.ai_list:
            mrks_inv.logger.info(f"\nUse net, pos: {iatom}, atom type: {atom}")
            if args.check:
                mrks_inv.logger.info(
                    f"\nUse data for check, pos: {iatom}, atom type: {atom}"
                )
                path_mask = (
                    Path(f"mrks-e-{args.molecular}-{atom}-{args.name}")
                    / "data"
                    / "masks"
                )
                file_mask = list(
                    (path_mask).glob(
                        f"data-{args.molecular}-*-{distance:.4f}-{iatom}.npy"
                    )
                )
                model_dict[f"{atom}-{iatom}-check"] = np.load(file_mask[0])
        else:
            mrks_inv.logger.info(f"\nUse data, pos: {iatom}, atom type: {atom}")
            path_mask = (
                Path(f"mrks-e-{args.molecular}-{atom}-{args.name}") / "data" / "masks"
            )
            file_mask = list(
                (path_mask).glob(f"data-{args.molecular}-*-{distance:.4f}-{iatom}.npy")
            )
            model_dict[f"{atom}-{iatom}"] = np.load(file_mask[0])

    for step in range(args.scf_step):
        dm1_r = mrks_inv.aux_function.oe_rho_r(dm1)
        if WEIGHT:
            dm1_r_grid = mrks_inv.grids.vector_to_matrix(dm1_r * mrks_inv.grids.weights)
        else:
            dm1_r_grid = mrks_inv.grids.vector_to_matrix(dm1_r)

        vxc_grid = np.zeros_like(dm1_r_grid)

        for iatom, rho_0_grid_atom in enumerate(dm1_r_grid):
            atom = mrks_inv.mol.atom[iatom][0]
            if atom in args.ai_list:
                vxc_grid[iatom] = predict_potential(
                    model_dict[atom],
                    rho_0_grid_atom[np.newaxis, :, :],
                    device,
                    args.if_pad,
                    args.if_flatten,
                    "sort" in args.name,
                )[0, 0, :, :]
                if args.check:
                    vxc_grid_check = model_dict[f"{atom}-{iatom}-check"][0, :, :]
                    mrks_inv.logger.info(f"\npredict vxc {numpy2str(vxc_grid)}\n")
                    mrks_inv.logger.info(f"\n{numpy2str(vxc_grid_check)}\n")
            else:
                vxc_grid[iatom] = model_dict[f"{atom}-{iatom}"][0, :, :]

        vxc = mrks_inv.grids.matrix_to_vector(vxc_grid)
        if XC_CODE is not None:
            lda_exc, lda_vxc = pyscf.dft.libxc.eval_xc(XC_CODE, dm1_r)[:2]
            lda_vxc = lda_vxc[0]
            vxc += lda_vxc
        xc_v = mrks_inv.aux_function.oe_fock(vxc, mrks_inv.grids.weights)
        vj = mrks_inv.myhf.get_jk(mrks_inv.mol, dm1, 1)[0]
        fock_a = mrks_inv.mat_hs @ (h1e + vj + xc_v) @ mrks_inv.mat_hs
        eigvecs, mo = np.linalg.eigh(fock_a)
        mo = mrks_inv.mat_hs @ mo
        dm1_old = dm1.copy()
        dm1 = 2 * mo[:, : mrks_inv.nocc] @ mo[:, : mrks_inv.nocc].T
        dm1 = mrks_inv.hybrid(dm1, dm1_old)
        error = np.linalg.norm(dm1 - dm1_old)
        if args.noisy_print:
            mrks_inv.logger.info(f"\nerror of dm1, {error:.4e}")
        else:
            if step % 100 == 0:
                mrks_inv.logger.info(f"\nerror of dm1, {error:.4e}")
            elif step % 10 == 0:
                mrks_inv.logger.info(".")

        if error < args.error_scf:
            break

    mrks_inv.logger.info("\nSCF DONE")

    ccsd_dm1_file = list(mrks_inv.path.glob("ccsd-dm1.npy"))
    ccsdt_dm1_file = list(mrks_inv.path.glob("ccsdt-dm1.npy"))
    if (
        (len(ccsd_dm1_file) == 1)
        and (len(ccsdt_dm1_file) == 1)
        and ("sort" not in args.name)
    ):
        ccsd_dm1 = np.load(ccsd_dm1_file[0])
        ccsdt_dm1 = np.load(ccsdt_dm1_file[0])
        with open(mrks_inv.path / "energy.json", "r", encoding="utf-8") as f:
            data_json = json.load(f)
            ccsd_e, ccsdt_e = data_json["e_cisd"], data_json["e_cisdt"]
    else:
        mo = myhf.mo_coeff
        mycc = pyscf.cc.CCSD(myhf).run()
        ccsd_e = mycc.e_tot
        ccsd_dm1_mo = mycc.make_rdm1()
        ccsd_dm1 = oe.contract("ij,pi,qj->pq", ccsd_dm1_mo, mo, mo)

        if "sort" not in args.name:
            np.save(mrks_inv.path / "ccsd-dm1.npy", ccsd_dm1)
        mrks_inv.logger.info("\nCCSD DONE")

        mycc = pyscf.cc.CCSD(myhf)
        mycc.conv_tol = 1e-12
        _, t1, t2 = mycc.kernel()
        eris = mycc.ao2mo()

        e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
        ccsdt_e = ccsd_e + e3ref
        l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
        ccsdt_dm1_mo = ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris)
        ccsdt_dm1 = oe.contract("ij,pi,qj->pq", ccsdt_dm1_mo, mo, mo)

        if "sort" not in args.name:
            with open(mrks_inv.path / "energy.json", "w", encoding="utf-8") as f:
                energy = {}
                energy["e_cisdt"] = ccsdt_e
                energy["e_cisd"] = ccsd_e
                json.dump(energy, f, indent=4)
            np.save(mrks_inv.path / "ccsdt-dm1.npy", ccsdt_dm1)
        mrks_inv.logger.info("\nCCSDT DONE")

    dm1_dft_r = mrks_inv.aux_function.oe_rho_r(dm1_dft)
    dm1_ccsd_r = mrks_inv.aux_function.oe_rho_r(ccsd_dm1)
    dm1_r = mrks_inv.aux_function.oe_rho_r(dm1)
    dm1_ccsdt_r = mrks_inv.aux_function.oe_rho_r(ccsdt_dm1)

    error_dft = np.sum(abs(dm1_dft_r - dm1_ccsdt_r) * mrks_inv.grids.weights)
    error_ccsd = np.sum(abs(dm1_ccsd_r - dm1_ccsdt_r) * mrks_inv.grids.weights)
    error_scf = np.sum(abs(dm1_r - dm1_ccsdt_r) * mrks_inv.grids.weights)

    save_data["error_dft"] = error_dft
    save_data["error_ccsd"] = error_ccsd
    save_data["error_scf"] = error_scf

    mrks_inv.logger.info(
        f"\nCheck!\n"
        f"{error_dft:16.10f}\n"
        f"{error_ccsd:16.10f}\n"
        f"{error_scf:16.10f}\n"
    )

    if WEIGHT:
        dm1_r_grid = mrks_inv.grids.vector_to_matrix(dm1_r * mrks_inv.grids.weights)
    else:
        dm1_r_grid = mrks_inv.grids.vector_to_matrix(dm1_r)
    exc_grid = np.zeros_like(dm1_r_grid)

    for iatom, rho_0_grid_atom in enumerate(dm1_r_grid):
        atom = mrks_inv.mol.atom[iatom][0]
        if atom in args.ai_list:
            exc_grid[iatom] = predict_potential(
                model_dict[atom],
                rho_0_grid_atom[np.newaxis, :, :],
                device,
                args.if_pad,
                args.if_flatten,
                "sort" in args.name,
            )[0, 1, :, :]
        else:
            exc_grid[iatom] = model_dict[f"{atom}-{iatom}"][1, :, :]

    exc = mrks_inv.grids.matrix_to_vector(exc_grid)
    if XC_CODE is not None:
        lda_exc, lda_vxc = pyscf.dft.libxc.eval_xc(XC_CODE, dm1_r)[:2]
        exc += lda_exc

    e_h1 = oe.contract("ij,ji->", mrks_inv.h1e, dm1)
    e_vj = oe.contract("pqrs,pq,rs->", mrks_inv.eri, dm1, dm1)
    ene_t_vc = (
        e_h1
        + mrks_inv.mol.energy_nuc()
        + 0.5 * e_vj
        + np.sum(exc * dm1_r * mrks_inv.grids.weights)
    )
    mrks_inv.logger.info(
        "\n"
        f"{(mrks_inv.au2kjmol * ccsdt_e):23.10f}\n"
        f"{(mrks_inv.au2kjmol * ccsd_e):23.10f}\n"
        f"{(mrks_inv.au2kjmol * e_dft):23.10f}\n"
        f"{(mrks_inv.au2kjmol * ene_t_vc):23.10f}\n"
    )

    save_data["e_cisdt"] = mrks_inv.au2kjmol * ccsdt_e
    save_data["e_cisd"] = mrks_inv.au2kjmol * ccsd_e
    save_data["e_dft"] = mrks_inv.au2kjmol * e_dft
    save_data["ene_t_vc"] = mrks_inv.au2kjmol * ene_t_vc

    with open(mrks_inv.path / "save_data.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4)

    del mrks_inv
    gc.collect()
    torch.cuda.empty_cache()
