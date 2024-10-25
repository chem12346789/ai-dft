import time
import subprocess
from pathlib import Path
import itertools
import sys

print(sys.argv)
main_dir = Path(__file__).resolve().parents[0]
template_bash = main_dir / "train-direct.bash"
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

# renew out_mkdir
if (main_dir / "out_mkdir").exists():
    (main_dir / "out_mkdir").unlink()
(main_dir / "out_mkdir").touch()

work_dir = main_dir / ("bash_submitted" + time_stamp)
work_dir.mkdir()
work_bash = work_dir / "train-direct.bash"

LIST_OF_GPU = itertools.cycle([0, 1])
GPU_NODE_POOL = itertools.cycle(
    [
        "gpu07",
    ]
)

for (
    batch_size,
    eval_step,
    input_size,
    hidden_size,
    output_size,
    num_layer,
    residual,
    load_model,
    (pot_weight, ene_weight),
    with_eval,
    precision,
) in itertools.product(
    [32],
    [10],
    [1],  # input_size
    [64],  # hidden_size
    [1],  # output_size
    [4],  # num_layer
    [-1],  # residual
    ["New"],  # load_model
    [
        (0, 0),
    ],
    [
        "True",
    ],
    [
        "float32",
        "float64",
    ],
):
    number_of_gpu = next(LIST_OF_GPU)
    cmd = f"""cp {template_bash} {work_bash}"""
    gpu_node = next(GPU_NODE_POOL)
    cmd += "&&" + f"""sed -i "s/BASH_GPU_NODE/{gpu_node}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/INPUT_SIZE/{input_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/HIDDEN_SIZE/{hidden_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/OUTPUT_SIZE/{output_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/EVAL_STEP/{eval_step}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BATCH_SIZE/{batch_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/NUM_LAYER/{num_layer}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/RESIDUAL/{residual}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/ENE_WEIGHT/{ene_weight}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/POT_WEIGHT/{pot_weight}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/WITH_EVAL/{with_eval}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/LOAD_MODEL/{load_model}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/NUMBER_OF_GPU/{number_of_gpu}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/PRECISION/{precision}/g" {work_bash}"""
    cmd += (
        "&&"
        + f"""mv {work_bash} {work_dir / f"train_{input_size}_{hidden_size}_{output_size}_{eval_step}_{batch_size}_{num_layer}_{residual}_{ene_weight}_{pot_weight}_{with_eval}_{load_model}-{precision}.bash"}"""
    )
    with open(main_dir / "out_mkdir", "w", encoding="utf-8") as f:
        subprocess.call(cmd, shell=True, stdout=f)

for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd = "nohup bash -c '"
        cmd += f"""bash {child} > log/{child.stem};"""
        cmd = cmd[:-1]
        cmd += "'&"

# print(cmd)

with open(f"log/out_{time_stamp}", "a", encoding="utf-8") as f:
    subprocess.call(cmd, shell=True, stdout=f)
time.sleep(0.01)
