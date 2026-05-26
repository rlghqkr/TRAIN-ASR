"""SenseVoice 학습 래퍼 — FunASR `train_ds.py` 를 `torchrun` 으로 호출.

참고: practice/2601_sensevoice_train/finetune.sh

JupyterLab 노트북에선 셀에서 *명령 출력* → 별도 터미널 실행 (tmux 권장).
백그라운드 실행은 `run_in_background()` 로 가능하지만 노트북 커널이 꺼지면 같이 죽음.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _funasr_train_tool() -> Path:
    try:
        import funasr
    except ImportError as e:
        raise ImportError("funasr 설치 필요: `pip install funasr`") from e
    return Path(funasr.__path__[0]) / "bin" / "train_ds.py"


def build_torchrun_command(cfg: dict[str, Any]) -> str:
    """config dict → `torchrun ... train_ds.py ...` 셸 명령 문자열.

    Args:
        cfg: configs/<exp>.yaml 의 dict.

    Returns:
        bash 에서 그대로 실행 가능한 명령 (멀티라인).
    """
    train_tool = _funasr_train_tool()

    gpus = cfg["runtime"].get("cuda_visible_devices", "0")
    gpu_num = len([x for x in gpus.split(",") if x.strip()])

    backbone = cfg["models"]["sensevoice"]["backbone"]
    train_data = cfg["paths"]["train_jsonl"]
    val_data = cfg["paths"]["val_jsonl"]
    output_dir = Path(cfg["paths"]["outputs_dir"]) / cfg["experiment"]["name"]

    tr = cfg["training"]
    batch_size = tr.get("batch_size_tokens", tr["batch_size"] * 1000)
    lr = tr["lr"]
    max_epoch = tr["epochs"]

    # SenseVoice 어댑터로 변환된 JSONL 권장 (별도 셀에서 to_sensevoice_jsonl 호출)
    sv_train = cfg.get("data", {}).get("sensevoice_train_jsonl", train_data)
    sv_val = cfg.get("data", {}).get("sensevoice_val_jsonl", val_data)

    cmd = (
        f'export CUDA_VISIBLE_DEVICES="{gpus}"\n'
        f"torchrun --nnodes 1 --nproc_per_node {gpu_num} --master_port 26669 "
        f"{train_tool} \\\n"
        f'    ++model="{backbone}" \\\n'
        f"    ++trust_remote_code=true \\\n"
        f'    ++train_data_set_list="{sv_train}" \\\n'
        f'    ++valid_data_set_list="{sv_val}" \\\n'
        f"    ++dataset_conf.batch_size={batch_size} \\\n"
        f'    ++dataset_conf.batch_type="token" \\\n'
        f"    ++train_conf.max_epoch={max_epoch} \\\n"
        f"    ++train_conf.log_interval=1 \\\n"
        f"    ++train_conf.validate_interval=2000 \\\n"
        f"    ++train_conf.save_checkpoint_interval=2000 \\\n"
        f"    ++optim_conf.lr={lr} \\\n"
        f'    ++output_dir="{output_dir}"'
    )
    return cmd


def run_in_background(cmd: str, *, log_path: str | Path) -> subprocess.Popen:
    """학습 명령을 백그라운드 subprocess 로 실행. 로그는 파일로.

    노트북 커널이 꺼지면 함께 죽으므로 장시간 학습은 **tmux + bash** 권장.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        stdout=log, stderr=subprocess.STDOUT,
    )
    return proc


def print_command(cmd: str) -> None:
    """노트북 셀 사용용 — 명령을 보기 좋게 출력."""
    print("# SenseVoice 학습 명령 (별도 터미널에서 실행 권장)")
    print(cmd)
