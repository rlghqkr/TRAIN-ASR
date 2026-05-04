"""학습 진입점 보일러플레이트.

설정·시드·W&B 연결 흐름의 예시입니다. 실제 학습 로직은 프로젝트에 맞게 채우세요.

사용법:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import structlog


# 프로젝트 루트를 import 경로에 추가 (editable install 안 했을 때 대비)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.utils import load_config, seed_everything  # noqa: E402
from project.utils.wandb_utils import finish_wandb, init_wandb  # noqa: E402


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
        "--run-name",
        type=str,
        default=None,
        help="Optional W&B run name. If omitted, wandb auto-generates one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    log = logger.bind(
        experiment=config["experiment"]["name"],
        config_path=args.config,
    )
    log.info("Loaded config")

    seed_everything(config["experiment"]["seed"])
    log.info("Seed set", seed=config["experiment"]["seed"])

    run = init_wandb(config, run_name=args.run_name)

    try:
        # ---------------------------------------------------------------------
        # TODO: 학습 로직을 채워 넣으세요.
        # ---------------------------------------------------------------------
        log.info("Training loop placeholder — implement here")

    except Exception as e:
        log.error("Training failed", error_type=type(e).__name__, error_message=str(e))
        raise
    finally:
        finish_wandb(run)
        log.info("Training finished")


if __name__ == "__main__":
    main()
