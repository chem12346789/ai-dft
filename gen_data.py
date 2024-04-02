import numpy as np
from pathlib import Path
import argparse
import json


def clean_dir(pth):
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            clean_dir(child)


parser = argparse.ArgumentParser(description="Obtain data from npy files")

parser.add_argument(
    "--name",
    "-n",
    type=str,
    default="First_Run",
    help="Witch directory we save data to.",
)

parser.add_argument(
    "--energy",
    "-e",
    type=bool,
    default=False,
    help="If contain energy.",
)

parser.add_argument(
    "--atom",
    type=str,
    default="H",
    help="Atom name. Default is 'H'",
)

parser.add_argument(
    "--data",
    "-d",
    nargs="+",
    type=str,
    default="data",
    help='Name of data directory. Could be a list. Default is "data".',
)

parser.add_argument(
    "--nclass",
    "-c",
    type=int,
    default=1,
    help="Number of nclass.",
)

args = parser.parse_args()

main_dir = Path(__file__).resolve().parents[1]
imgs_path = main_dir / args.name / "data" / "imgs"
mask_path = main_dir / args.name / "data" / "masks"
weit_path = main_dir / args.name / "data" / "weights"

imgs_path.mkdir(parents=True, exist_ok=True)
mask_path.mkdir(parents=True, exist_ok=True)
weit_path.mkdir(parents=True, exist_ok=True)
clean_dir(imgs_path)
clean_dir(mask_path)
clean_dir(weit_path)
print(f"data: {args.data}")

MESSAGE = ""

for data_i in args.data:
    data_path = Path("./") / data_i
    for child in (data_path).glob("*"):
        if not child.is_dir():
            continue
        print(f"Processing {child.parts[-1]}")
        if "small" in args.name:
            if (
                np.abs(int(float(child.parts[-1]) * 20) * 0.05 - float(child.parts[-1]))
                > 1e-3
            ):
                continue
        
        # exclude the data with long distance.
        if float(child.parts[-1]) > 3.501:
            continue
        
        data_file = list(child.glob("rho_inv_mrks.npy"))
        masks_v_file = list(child.glob("mrks.npy"))
        lda_v_file = list(child.glob("lda.npy"))
        weight_file = list(child.glob("weight.npy"))

        if (
            (len(data_file) == 1)
            and (len(masks_v_file) == 1)
            and (len(lda_v_file) == 1)
            and (len(weight_file) == 1)
        ):
            data = np.load(data_file[0])
            masks_v = np.load(masks_v_file[0])
            weight = np.load(weight_file[0])

            if "weit" in args.name:
                data_weit = data * weight
                masks_v_weit = masks_v.copy()
            else:
                data_weit = data.copy()
                masks_v_weit = masks_v.copy()

            if "sort" in args.name:
                data_sort_index = np.zeros_like(data, dtype=int)
                data_sort_back_index = np.zeros_like(data, dtype=int)
                data_sort = np.zeros_like(data_weit, dtype=float)
                masks_v_weit_sort = np.zeros_like(masks_v_weit, dtype=float)
                for i in range(data_weit.shape[0]):
                    for j in range(data_weit.shape[1]):
                        data_sort_index[i, j, :] = np.argsort(
                            np.sum(data_weit[i, :, :], axis=0)
                        )
                        for k in range(data_weit.shape[2]):
                            data_sort_back_index[i, j, data_sort_index[i, j, k]] = k
                        data_sort[i, j, :] = data_weit[i, j, data_sort_index[i, j, :]]
                        masks_v_weit_sort[i, j, :] = masks_v_weit[
                            i, j, data_sort_index[i, j, :]
                        ]
                masks_v_weit = masks_v_weit_sort.copy()
                data_weit = data_sort.copy()

            if args.energy:
                masks_e_file = list(child.glob("mrks_e.npy"))
                lda_e_file = list(child.glob("lda_e.npy"))
                masks_tr_file = list(child.glob("tr.npy"))
                masks_e = (np.load(masks_e_file[0]) + np.load(masks_tr_file[0])) / (
                    data + 1e-14
                )

                if "weit" in args.name:
                    masks_e_weit = masks_e.copy()
                else:
                    masks_e_weit = masks_e.copy()

                if "sort" in args.name:
                    masks_e_weit_sort = np.zeros_like(masks_e_weit, dtype=float)
                    for i in range(data_weit.shape[0]):
                        for j in range(data_weit.shape[1]):
                            masks_e_weit_sort[i, j, :] = masks_e_weit[
                                i, j, data_sort_index[i, j, :]
                            ]
                    masks_e_weit = masks_e_weit_sort.copy()

            with open(child / "mol_info.json", "r", encoding="utf-8") as f:
                mol_info = json.load(f)

            for i in range(data.shape[0]):
                if mol_info["atom"][i][0] != args.atom:
                    continue

                data_name = f"{data_path.parts[-1]}-{child.parts[-1]}-{i}.npy"
                masks_ve = np.zeros((1, data.shape[1], data.shape[2]))
                masks_ve[0, :, :] = masks_v[i, :, :]

                if args.energy:
                    if args.nclass == 1:
                        if len(masks_e_file) == 1:
                            masks_ve[0, :, :] = masks_e[i, :, :]
                    elif args.nclass == 2:
                        masks_ve = np.zeros((2, data.shape[1], data.shape[2]))
                        if len(masks_e_file) == 1:
                            masks_ve[0, :, :] = masks_v[i, :, :]
                            masks_ve[1, :, :] = masks_e[i, :, :]

                data_shape = (1, data.shape[1], data.shape[2])
                np.save(imgs_path / data_name, data_weit[i, :, :].reshape(data_shape))
                np.save(mask_path / data_name, masks_ve)
                np.save(weit_path / data_name, weight[i, :, :].reshape(data_shape))
                if "sort" in args.name:
                    sort_path = main_dir / args.name / "data" / "sort"
                    sort_back_path = main_dir / args.name / "data" / "sort_back"
                    sort_path.mkdir(parents=True, exist_ok=True)
                    sort_back_path.mkdir(parents=True, exist_ok=True)
                    clean_dir(weit_path)
                    clean_dir(sort_path)
                    np.save(sort_path / data_name, data_sort_index[i, :, :])
                    np.save(sort_back_path / data_name, data_sort_back_index[i, :, :])

                print(
                    f"""{child.parts[-1]} max of imag {np.max(data_weit[i, :, :]):.3e} """
                    f"""min of imag {np.min(data_weit[i, :, :]):.3e}"""
                )
                if args.energy:
                    print(
                        f"""{child.parts[-1]} max of masks_e {np.max(masks_e[i, :, :]):.3e} """
                        f"""min of masks {np.min(masks_e[i, :, :]):.3e}"""
                    )
                print(
                    f"""{child.parts[-1]} max of masks_v {np.max(masks_v[i, :, :]):.3e} """
                    f"""min of masks {np.min(masks_v[i, :, :]):.3e}"""
                )
        else:
            MESSAGE += f"""{child.parts[-1]} not found\n"""
print("\n")
print(MESSAGE)
