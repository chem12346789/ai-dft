import sys
import time
import subprocess
import numpy as np
from pathlib import Path


main_dir = Path(__file__).resolve().parents[0]
template_dir = main_dir / "template"
len_template = len(list(template_dir.glob("*")))

# renew out_mkdir
if (main_dir / "out_submit").exists():
    (main_dir / "out_submit").unlink()
(main_dir / "out_submit").touch()

for child in (main_dir / "data").glob("*"):
    if not (child.is_file()):
        # if len(list(child.glob("*"))) == len_template:
        if len(list(child.glob("mrks.npy"))) == 0:
            print(f"""submit {child}""")
            batch_file = child / "deom.bash"
            if batch_file.exists():
                cmd = f"""cd {child}"""
                cmd += "&&" + "sbatch < deom.bash"
                with open(main_dir / "out_submit", "a") as outfile:
                    result = subprocess.call(cmd, shell=True, stdout=outfile)
        else:
            print(f"""{child} not complete or have been changed/submitted""")
