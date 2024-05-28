import pandas as pd
import numpy as np


def save_csv_loss(dice_after_train, path):
    """
    save the loss to a csv file
    """
    data_frame = {}
    data_frame["mean"] = []
    for i_str in dice_after_train[-1]:
        data_frame[i_str] = []

    for j, _ in enumerate(dice_after_train[:-1]):
        data_frame["mean"].append(np.mean(np.abs(dice_after_train[j])))
        for i, i_str in enumerate(dice_after_train[-1]):
            data_frame[i_str].append(dice_after_train[j][i])
    pd.DataFrame(data=data_frame).to_csv(path)
