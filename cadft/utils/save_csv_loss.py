import pandas as pd
import numpy as np


def save_csv_loss(dice_after_train, path):
    """
    save the loss to a csv file
    """
    data_frame = {}
    data_frame["mean"] = [
        np.mean(dice_after_train[0]),
        np.mean(dice_after_train[1]),
        np.mean(dice_after_train[2]),
        np.mean(dice_after_train[3]),
        np.mean(dice_after_train[4]),
        np.mean(dice_after_train[5]),
    ]
    for i, i_str in enumerate(dice_after_train[5]):
        data_frame[i_str] = [
            dice_after_train[0][i],
            dice_after_train[1][i],
            dice_after_train[2][i],
            dice_after_train[3][i],
            dice_after_train[4][i],
            dice_after_train[5][i],
        ]
    pd.DataFrame(data=data_frame).to_csv(path)
