"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

MAIN_PATH = os.environ.get("AIDFT_MAIN_PATH")
if MAIN_PATH is None:
    MAIN_PATH = Path(__file__).parent.parent.parent
else:
    MAIN_PATH = Path(MAIN_PATH)

DATA_PATH = os.environ.get("DATA_PATH")
if DATA_PATH is None:
    DATA_PATH = MAIN_PATH / "data" / "grids_mrks"
else:
    DATA_PATH = Path(DATA_PATH)

DATA_SAVE_PATH = os.environ.get("DATA_SAVE_PATH")
if DATA_SAVE_PATH is None:
    DATA_SAVE_PATH = MAIN_PATH / "data" / "grids_mrks" / "saved_data"
else:
    DATA_SAVE_PATH = Path(DATA_SAVE_PATH)

DATA_CC_PATH = os.environ.get("DATA_SAVE_PATH")
if DATA_CC_PATH is None:
    DATA_CC_PATH = MAIN_PATH / "data" / "test"
else:
    DATA_CC_PATH = Path(DATA_CC_PATH)

CHECKPOINTS_PATH = os.environ.get("CHECKPOINTS_PATH")
if CHECKPOINTS_PATH is None:
    CHECKPOINTS_PATH = MAIN_PATH / "checkpoints"
else:
    CHECKPOINTS_PATH = Path(CHECKPOINTS_PATH)

print(f"MAIN_PATH: {MAIN_PATH.resolve()}")
print(f"DATA_PATH: {DATA_PATH.resolve()}")
print(f"DATA_SAVE_PATH: {DATA_SAVE_PATH.resolve()}")
print(f"DATA_CC_PATH: {DATA_CC_PATH.resolve()}")

if __name__ == "__main__":
    print(MAIN_PATH)
    print(DATA_PATH)
    print(DATA_SAVE_PATH)
    print(DATA_CC_PATH)
