"""YAML 설정 로더.

argparse + YAML 조합으로 단순하게 실험 설정 관리.
"""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """설정 파일 로딩/검증 관련 예외."""


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 로드해서 dict로 반환.

    Args:
        path: YAML 파일 경로.

    Returns:
        파싱된 설정 dict.

    Raises:
        ConfigError: 파일이 없거나 YAML 파싱 실패 시.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config: {config_path}") from e

    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a mapping at top level: {config_path}")

    return config


def get_required(config: dict[str, Any], key: str) -> Any:
    """필수 설정 키를 명시적으로 가져옴 (없으면 즉시 에러).

    Fail-fast 원칙에 따라 기본값 fallback을 제공하지 않음.
    중첩 키는 점(.)으로 표현: get_required(cfg, "training.learning_rate")
    """
    keys = key.split(".")
    current: Any = config
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            raise ConfigError(f"Required config key missing: {key}")
        current = current[k]
    return current
