"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

MAIN_PATH = os.environ.get("AIDFT_MAIN_PATH")
if MAIN_PATH is None:
    MAIN_PATH = Path(__file__).parent.parent.parent

DATA_PATH = os.environ.get("DATA_PATH")
if DATA_PATH is None:
    DATA_PATH = MAIN_PATH / "data" / "grids_mrks"

DATA_SAVE_PATH = os.environ.get("DATA_SAVE_PATH")
if DATA_SAVE_PATH is None:
    DATA_SAVE_PATH = MAIN_PATH / "data" / "grids_mrks" / "saved_data"

DATA_CC_PATH = os.environ.get("DATA_SAVE_PATH")
if DATA_CC_PATH is None:
    DATA_CC_PATH = MAIN_PATH / "data" / "test"

if __name__ == "__main__":
    print(MAIN_PATH)
    print(DATA_PATH)
    print(DATA_SAVE_PATH)
    print(DATA_CC_PATH)
