"""Weights & Biases 헬퍼.

설정 파일 기반으로 wandb를 일관되게 초기화/종료.
"""

from typing import Any


class WandbInitError(Exception):
    """W&B 초기화 관련 예외."""


def init_wandb(config: dict[str, Any], *, run_name: str | None = None) -> Any:
    """설정 dict 기반으로 W&B run 초기화.

    Args:
        config: 전체 실험 설정. `wandb.enabled`, `wandb.project`, `wandb.entity` 등을 사용.
        run_name: 명시적인 run 이름. 미지정 시 wandb가 자동 생성.

    Returns:
        wandb.Run 객체. `wandb.enabled=false` 인 경우 None.

    Raises:
        WandbInitError: wandb 미설치 또는 필수 설정 누락 시.
    """
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as e:
        raise WandbInitError(
            "wandb is not installed. Install with: pip install wandb"
        ) from e

    project = wandb_cfg.get("project")
    if not project:
        raise WandbInitError("wandb.project is required when wandb.enabled is true")

    run = wandb.init(
        project=project,
        entity=wandb_cfg.get("entity"),
        name=run_name,
        tags=wandb_cfg.get("tags", []),
        notes=wandb_cfg.get("notes", ""),
        config=config,
    )
    return run


def finish_wandb(run: Any) -> None:
    """W&B run 명시적으로 종료. run이 None이면 무시."""
    if run is None:
        return
    try:
        import wandb

        wandb.finish()
    except ImportError:
        pass
