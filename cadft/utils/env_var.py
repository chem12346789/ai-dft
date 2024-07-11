"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

MAIN_PATH = os.environ.get("AIDFT_MAIN_PATH")
if MAIN_PATH is None:
    MAIN_PATH = Path(__file__).parent.parent.parent

DATA_MAIN_PATH = os.environ.get("AIDFT_MAIN_PATH")
if MAIN_PATH is None:
    DATA_MAIN_PATH = MAIN_PATH / "data" / "grids_mrks"

if __name__ == "__main__":
    print(MAIN_PATH)
    print(DATA_MAIN_PATH)
