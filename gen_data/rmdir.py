import sys
import time
import subprocess
import numpy as np
from pathlib import Path


def clean_dir(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            clean_dir(child)
            child.rmdir()


main_dir = Path(__file__).resolve().parents[0]
for child in (main_dir / 'data').glob('*'):
    if not (child.is_file()):
        clean_dir(child)
        child.rmdir()
