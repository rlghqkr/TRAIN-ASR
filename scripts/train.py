"""학습 진입점 (CLI).

config·시드 셋업 후 모델별 학습으로 디스패치한다. 실제 학습 로직은
project/training/run.py 에 있고, 노트북도 같은 함수를 호출한다(SoT).

사용법:
    # Whisper — HF Trainer 로 바로 학습
    python scripts/train.py --config configs/default.yaml --model whisper

    # SenseVoice — torchrun 셸 명령만 출력 (별도 tmux 에서 실행)
    python scripts/train.py --config configs/default.yaml --model sensevoice
"""

import os

# 오디오 변환(feature extraction)이 PyTorch intra-op 으로 192 코어를 다 잡아 서버 load 폭주 +
# 같은 서버 이웃에게 민폐 (실제 병목은 디스크라 변환 속도엔 영향 없음). torch/numpy import 전에
# 스레드 풀을 제한해 코어 독식을 막는다(dataloader 워커도 이 환경을 상속). 명시 지정 시 그걸 존중.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import structlog


# 프로젝트 루트를 import 경로에 추가 (editable install 안 했을 때 대비)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.utils import load_config, seed_everything  # noqa: E402


logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/default.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["whisper", "sensevoice"],
        help="학습할 모델 종류 (명시 필수).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional W&B run name. If omitted, experiment.name 을 사용.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    log = logger.bind(
        experiment=config["experiment"]["name"],
        model=args.model,
        config_path=args.config,
    )
    log.info("Loaded config")

    seed_everything(config["experiment"]["seed"])
    log.info("Seed set", seed=config["experiment"]["seed"])

    try:
        if args.model == "whisper":
            # Whisper: HF Trainer 가 W&B 까지 직접 다룬다 (report_to=wandb).
            from project.training.run import run_whisper_training

            result = run_whisper_training(config, run_name=args.run_name)
            log.info("Whisper training done", output_dir=result["output_dir"])

        elif args.model == "sensevoice":
            # SenseVoice: torchrun 외부 프로세스라 여기선 명령만 출력.
            from project.training.sensevoice import (
                build_torchrun_command,
                print_command,
            )

            cmd = build_torchrun_command(config)
            print_command(cmd)
            log.info("SenseVoice 는 위 torchrun 명령을 tmux 에서 직접 실행하세요")

    except Exception as e:
        log.error("Training failed", error_type=type(e).__name__, error_message=str(e))
        raise


if __name__ == "__main__":
    main()
