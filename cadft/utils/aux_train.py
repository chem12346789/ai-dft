"""
Generate list of keys for the dictionary that will store the model.
"""

from itertools import product
from pathlib import Path

import torch

from cadft.utils.nao import NAO
from cadft.utils.model.fc_net import FCNet as Model

# from cadft.utils.model.transformer import Transformer as Model


def gen_keys_l(atom_list):
    """
    input:
        atom_list: list of atom names
    output:
        keys: 1st and 2nd words are atom names, 3rd is if diagonal (H-H-O or H-H-D)
    """
    keys_l = []

    for i_atom, j_atom in product(atom_list, atom_list):
        if i_atom != j_atom:
            atom_name = f"{i_atom}-{j_atom}"
            keys_l.append(atom_name)
        else:
            atom_name = f"{i_atom}-{i_atom}-D"
            keys_l.append(atom_name)
            atom_name = f"{i_atom}-{i_atom}-O"
            keys_l.append(atom_name)
    return keys_l


def gen_model_dict(keys_l, args, device):
    """
    input:
        keys: keys of dictionary, generated by gen_keys_l
    output:
        model_dict: dictionary of models
    """
    model_dict = {}

    for key in keys_l:
        model_dict[key + "1"] = Model(
            NAO[key.split("-")[0]] * NAO[key.split("-")[1]],
            args.hidden_size,
            NAO[key.split("-")[0]] * NAO[key.split("-")[1]],
        ).to(device)
        model_dict[key + "1"].double()

        model_dict[key + "2"] = Model(
            NAO[key.split("-")[0]] * NAO[key.split("-")[1]],
            args.hidden_size,
            1,
        ).to(device)
        model_dict[key + "2"].double()

    return model_dict


def load_model(model_dict, keys_l, args, device):
    """
    Load the model from the checkpoint.
    """
    if args.load != "":
        dir_load = Path(f"./checkpoint-{args.load}-{args.hidden_size}/")
        for key in keys_l:
            for i_str in ["1", "2"]:
                key_i_str = key + i_str
                list_of_path = list(dir_load.glob(f"{key}-{i_str}*.pth"))
                if len(list_of_path) == 0:
                    print(f"No model found for {key_i_str}, use random initialization.")
                    continue
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                state_dict = torch.load(load_path, map_location=device)
                model_dict[key_i_str].load_state_dict(state_dict)
                print(f"Model loaded from {load_path}")
