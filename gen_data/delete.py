import sys
import time
import subprocess
import numpy as np
from pathlib import Path

main_dir = Path(__file__).resolve().parents[0]

with open(main_dir / 'out_submit', "r") as outfile:
    for line in outfile:
        cmd = r'scancel {}'.format(line.split()[-1])
        with open('out', "w") as outfile:
            result = subprocess.call(cmd, shell=True, stdout=outfile)
