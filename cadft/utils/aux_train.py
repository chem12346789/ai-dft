"""
Generate list of keys for the dictionary that will store the model.
"""

from itertools import product
from pathlib import Path

import torch

from cadft.utils.nao import NAO
from cadft.utils.model.fc_net import FCNet as Model

# from cadft.utils.model.transformer import Transformer as Model


def gen_model_dict(atom_list, hidden_size, device):
    """
    input:
        keys: keys of dictionary, generated by gen_keys_l
    output:
        model_dict: dictionary of models
    """
    model_dict = {}

    for atom in atom_list:
        model_dict[atom + "1"] = Model(75, hidden_size, 75).to(device)
        model_dict[atom + "1"].double()
        model_dict[atom + "2"] = Model(75, hidden_size, 75).to(device)
        model_dict[atom + "2"].double()

    return model_dict


def load_model(model_dict, atom_list, load, hidden_size, device):
    """
    Load the model from the checkpoint.
    """
    if load != "":
        dir_load = Path(f"checkpoints/checkpoint-ccdft-{load}-{hidden_size}/")
        for atom in atom_list:
            for i_str in ["1", "2"]:
                key_i_str = atom + i_str
                list_of_path = list(dir_load.glob(f"{atom}-{i_str}*.pth"))
                if len(list_of_path) == 0:
                    print(f"No model found for {key_i_str}, use random initialization.")
                    continue
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                state_dict = torch.load(load_path, map_location=device)
                model_dict[key_i_str].load_state_dict(state_dict)
                print(f"Model loaded from {load_path}")
