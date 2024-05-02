import sys
import time
import subprocess
import numpy as np
from pathlib import Path

level = 4


def clean_dir(pth):
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            clean_dir(child)
            child.rmdir()


main_dir = Path(__file__).resolve().parents[0]
template_dir = main_dir / "template"

# renew out_mkdir
if (main_dir / "out_mkdir").exists():
    (main_dir / "out_mkdir").unlink()
(main_dir / "out_mkdir").touch()

for j in range(1, 2):
    for i_file in np.linspace(0.4, 0.9, 51):
        work_dir = main_dir / "data" / f"sample-{j:d}-{i_file:.3f}"

        if (work_dir).exists():
            print("skip", work_dir)
        else:
            # copy template to write-{:d}
            cmd = f"""cp -r {template_dir} {work_dir}"""
            cmd += (
                "&&"
                + f"""sed -i "s/DISTANCES_REPLACE/{i_file:.3f}/g" {work_dir/"deom.bash"}"""
            )
            cmd += (
                "&&"
                + f"""sed -i "s/LEVEL_REPLACE/{level:d}/g" {work_dir/"deom.bash"}"""
            )
            with open(main_dir / "out_mkdir", "w", encoding="utf-8") as outfile:
                subprocess.call(cmd, shell=True, stdout=outfile)
