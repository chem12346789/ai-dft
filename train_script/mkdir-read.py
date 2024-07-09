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
template_bash = main_dir / "train-template.bash"
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

critical_time = arrow.now().shift(hours=-72)
for item in Path(main_dir).glob("*"):
    if not item.is_file():
        ITEM_TIME = arrow.get(item.stat().st_mtime)
        if ITEM_TIME < critical_time:
            # remove it
            print(str(item.absolute()))
            clean_dir(item)
            item.rmdir()

# renew out_mkdir
if (main_dir / "out_mkdir").exists():
    (main_dir / "out_mkdir").unlink()
(main_dir / "out_mkdir").touch()

work_dir = main_dir / ("bash_submitted" + time_stamp)
work_dir.mkdir()
work_bash = work_dir / "train-template.bash"

input_size, output_size = 1, 1
# input_size, output_size = 302, 302

for (
    batch_size,
    eval_step,
    hidden_size,
    num_layer,
    residual,
    ene_weight,
    with_eval,
) in itertools.product(
    [64],
    [100],
    [64],
    [4],
    [0],
    [0, 1],
    [
        # "False",
        "True",
    ],
):
    cmd = f"""cp {template_bash} {work_bash}"""
    cmd += "&&" + f"""sed -i "s/INPUT_SIZE/{input_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/HIDDEN_SIZE/{hidden_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/OUTPUT_SIZE/{output_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/EVAL_STEP/{eval_step}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BATCH_SIZE/{batch_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/NUM_LAYER/{num_layer}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/RESIDUAL/{residual}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/ENE_WEIGHT/{ene_weight}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/WITH_EVAL/{with_eval}/g" {work_bash}"""
    cmd += (
        "&&"
        + f"""mv {work_bash} {work_dir / f"train_{hidden_size}_{eval_step}_{batch_size}_{num_layer}_{residual}_{ene_weight}.bash"}"""
    )
    with open(main_dir / "out_mkdir", "w", encoding="utf-8") as f:
        subprocess.call(cmd, shell=True, stdout=f)

for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd = f"""sbatch < {child}"""
        with open(main_dir / "out_mkdir", "a", encoding="utf-8") as f:
            subprocess.call(cmd, shell=True, stdout=f)

        # Best time for ai training is 6 seconds (according to the HuaWei)
        time.sleep(6)
