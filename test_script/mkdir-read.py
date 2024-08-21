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
template_bash = main_dir / "validate-template.bash"
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
work_bash = work_dir / "validate-template.bash"

for (checkpoint_hidden_size,) in itertools.product(
    [
        # "checkpoint-ccdft_2024-07-28-16-00-16_4_64_1_4_0",
        # "checkpoint-ccdft_2024-08-19-14-02-29_4_128_1_3_-1",
        # "checkpoint-ccdft_2024-08-19-15-22-55_4_128_2_3_-1",
        # "checkpoint-ccdft_2024-08-19-15-24-53_4_128_2_4_-1",
        # "checkpoint-ccdft_2024-08-19-18-57-18_4_128_2_4_-1",
        # "checkpoint-ccdft_2024-08-19-19-07-21_4_128_2_3_-1",
        # "checkpoint-ccdft_2024-08-19-19-13-32_4_128_2_3_-1",
        # "checkpoint-ccdft_2024-08-20-18-16-39_4_64_1_4_-1",
        # "checkpoint-ccdft_2024-08-20-18-19-49_1_64_-1_4_-1",
        "checkpoint-ccdft_2024-08-21-15-38-48_4_128_1_4_0",
        # "checkpoint-ccdft_2024-08-21-18-06-13_1_16_-1_3_0",
    ],
):
    (
        _,
        checkpoint,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        residual,
    ) = checkpoint_hidden_size.split("_")
    print(
        checkpoint,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        residual,
    )
    cmd = f"""cp {template_bash} {work_bash}"""
    cmd += "&&" + f"""sed -i "s/CHECKPOINT/{checkpoint}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/INPUT_SIZE/{input_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/HIDDEN_SIZE/{hidden_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/OUTPUT_SIZE/{output_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/NUM_LAYER/{num_layers}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/RESIDUAL/{residual}/g" {work_bash}"""
    cmd += (
        "&&"
        + f"""mv {work_bash} {work_dir / f"validate_{checkpoint_hidden_size}.bash"}"""
    )
    with open(main_dir / "out_mkdir", "w", encoding="utf-8") as f:
        subprocess.call(cmd, shell=True, stdout=f)

for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd = f"""sbatch < {child}"""
        with open(main_dir / "out_mkdir", "a", encoding="utf-8") as f:
            subprocess.call(cmd, shell=True, stdout=f)
        time.sleep(1)
