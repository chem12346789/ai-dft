import pandas as pd
import numpy as np

from cadft.utils.basis import gen_basis
from cadft.utils.rotate import rotate
from cadft.utils.parser import add_args
from cadft.utils.mrks import mrks, mrks_append
from cadft.utils.umrks import umrks
from cadft.utils.mrks_diis import mrks_diis
from cadft.utils.umrks_diis import umrks_diis
from cadft.utils.gmrks_diis import gmrks_diis
from cadft.utils.DataBase import gen_logger, process_input

from cadft.utils.DataBase import DataBase
from cadft.utils.model.fc_net import FCNet
from cadft.utils.model.transformer import Transformer
from cadft.utils.Grids import Grid
from cadft.utils.ModelDict import ModelDict
from cadft.utils.diis import DIIS

from cadft.utils.mol import Mol
from cadft.utils.env_var import MAIN_PATH, DATA_PATH, DATA_SAVE_PATH, DATA_CC_PATH


def save_csv_loss(name_list, path, loss_rho, loss_ene, loss_tot_ene):
    """
    save the loss to a csv file
    """
    df = pd.DataFrame(
        {
            "name": name_list,
            "loss_rho": loss_rho,
            "loss_ene": loss_ene,
            "loss_tot_ene": loss_tot_ene,
        }
    )
    df.to_csv(path, index=False)


NAO = {
    "H": 5,
    "C": 14,
}
