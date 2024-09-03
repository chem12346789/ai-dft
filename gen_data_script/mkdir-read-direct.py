import time
import subprocess
from pathlib import Path
import itertools
import sys

print(sys.argv)
main_dir = Path(__file__).resolve().parents[0]
template_bash = main_dir / "gen_data_template_direct.bash"
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

# renew out_mkdir
if (main_dir / "out_mkdir").exists():
    (main_dir / "out_mkdir").unlink()
(main_dir / "out_mkdir").touch()

work_dir = main_dir / ("bash_submitted" + time_stamp)
work_dir.mkdir()
work_bash = work_dir / "gen_data_template_direct.bash"

number_of_gpu = sys.argv[1] if len(sys.argv) > 1 else 0

for mol, basis_set, range_list, extend_atom in itertools.product(
    [
        # "methane",
        "ethane",
        "ethylene",
        "acetylene",
        # "propane",
        # "cyclopropane",
        # "cyclopropene",
        # "propylene",
        # "allene",
        # "methyl-openshell",
        # "ethyl-openshell",
    ],
    ["cc-pCVTZ"],
    [
        (-0.5, 2.5, 31),
        # (2.4, 2.5, 2),
        # (1.5, 2.0, 6),
    ],
    ["0-2"],
):
    cmd = f"""cp {template_bash} {work_bash}"""
    cmd += "&&" + f"""sed -i "s/MOL/{mol}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASIS/{basis_set}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/NUMBER_OF_GPU/{number_of_gpu}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/EXTEND_ATOM/{extend_atom}/g" {work_bash}"""

    if isinstance(range_list, float):
        start = range_list
        cmd += "&&" + f"""sed -i "s/START/{start}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/END//g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/STEP//g" {work_bash}"""
        cmd += (
            "&&"
            + f"""mv {work_bash} {work_dir / f"gen_data_{mol}_{basis_set}_{start}.bash"}"""
        )
    elif isinstance(range_list, tuple):
        start = range_list[0]
        end = range_list[1]
        step = range_list[2]
        cmd += "&&" + f"""sed -i "s/START/{start}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/END/{end}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/STEP/{step}/g" {work_bash}"""
        cmd += (
            "&&"
            + f"""mv {work_bash} {work_dir / f"gen_data_{mol}_{basis_set}_{start}_{end}_{step}.bash"}"""
        )
    with open(main_dir / "out_mkdir", "w", encoding="utf-8") as f:
        subprocess.call(cmd, shell=True, stdout=f)

cmd = "nohup bash -c '"
for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd += f"""bash {child} > log/{child.stem};"""
cmd = cmd[:-1]
cmd += "'&"

# print(cmd)

with open(f"log/out_{time_stamp}", "a", encoding="utf-8") as f:
    subprocess.call(cmd, shell=True, stdout=f)
time.sleep(0.01)
