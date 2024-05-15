import time
import subprocess
from pathlib import Path
import itertools
import arrow


def clean_dir(pth):
    """
    clean the directory
    """
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            clean_dir(child)
            child.rmdir()


main_dir = Path(__file__).resolve().parents[0]
template_bash = main_dir / "gen_data_template.bash"
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

# renew out_mkdir
if (main_dir / "out_mkdir").exists():
    (main_dir / "out_mkdir").unlink()
(main_dir / "out_mkdir").touch()

work_dir = main_dir / ("bash_submitted" + time_stamp)
work_dir.mkdir()
work_bash = work_dir / "gen_data_template.bash"

critical_time = arrow.now().shift(hours=-1)
for item in Path(main_dir).glob("*"):
    if not item.is_file():
        ITEM_TIME = arrow.get(item.stat().st_mtime)
        if ITEM_TIME < critical_time:
            print(str(item.absolute()))
            # remove it
            clean_dir(item)
            item.rmdir()

for mol, basis_set, range_list in itertools.product(
    [
        # "Methane",
        # "Ethane",
        # "Ethylene",
        # "Acetylene",
        "Allene",
        # "Propane",
        # "Propyne",
        "Cyclopropene",
        "Cyclopropane",
        # "Propylene",
        # "Isobutane",
        # "Butane",
        # "Butadiene",
        # "Butyne",
        "Bicyclobutane",
        "Cyclobutane",
        "Spiropentane",
        "Cyclopropylmethyl",
        # "Benzene",
        # "Neopentane",
        # "Cyclopentane",
        # "Pentane",
        # "Isopentane",
    ],
    ["cc-pvdz"],
    [
        (-0.5, 0.5, 51),
    ],
):
    cmd = f"""cp {template_bash} {work_bash}"""
    cmd += "&&" + f"""sed -i "s/MOL/{mol}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASIS/{basis_set}/g" {work_bash}"""
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

for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd = f"""sbatch < {child}"""
        with open(main_dir / "out_mkdir", "a", encoding="utf-8") as f:
            subprocess.call(cmd, shell=True, stdout=f)
        time.sleep(0.01)
