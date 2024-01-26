"""@package docstring
Documentation for this module.
 
More details.
"""
import logging
import numpy as np
from pathlib import Path


def gen_logger(distance_list, magic_str, path):
    """
    Function to distance list and generate logger
    """
    logger = logging.getLogger(__name__)
    logging.StreamHandler.terminator = ""
    if len(distance_list) == 3:
        distance_l = np.linspace(
            distance_list[0], distance_list[1], int(distance_list[2])
        )
        path_dir = path / f"data-{magic_str}"
        if not path_dir.exists():
            path_dir.mkdir(parents=True)
        Path(
            path_dir
            / f"inv_{distance_list[0]}_{distance_list[1]}_{distance_list[2]}.log"
        ).unlink(missing_ok=True)
        logger.addHandler(
            logging.FileHandler(
                path_dir
                / f"inv_{distance_list[0]}_{distance_list[1]}_{distance_list[2]}.log"
            )
        )
    else:
        distance_l = args.distance
        path_dir = path / f"data-{magic_str}"
        if not path_dir.exists():
            path_dir.mkdir(parents=True)
        Path(path_dir / "inv.log").unlink(missing_ok=True)
        logger.addHandler(logging.FileHandler(path_dir / f"inv.log"))
    logger.setLevel(logging.DEBUG)
    return distance_l, logger, path_dir
