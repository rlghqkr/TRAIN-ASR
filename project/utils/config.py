"""YAML 설정 로더.

argparse + YAML 조합으로 단순하게 실험 설정 관리.
"""

import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """설정 파일 로딩/검증 관련 예외."""


# `${a.b.c}` 형태의 보간 토큰. configs/default.yaml 의 paths 섹션처럼
# 다른 키 값을 참조하는 데 쓴다. omegaconf 의존 없이 표준 라이브러리로 해석.
_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


def _lookup(config: dict[str, Any], dotted: str) -> Any:
    """`a.b.c` 점 경로로 config 값을 찾는다. 없으면 Fail Fast."""
    current: Any = config
    for k in dotted.split("."):
        if not isinstance(current, dict) or k not in current:
            raise ConfigError(
                f"Config 보간 '${{{dotted}}}' 해석 실패: '{k}' 키 없음"
            )
        current = current[k]
    return current


def _resolve_interpolations(config: dict[str, Any]) -> dict[str, Any]:
    """`${a.b.c}` 참조를 실제 값으로 치환한 새 dict 를 반환.

    체인 참조(`gold_dir` → `data_root` → `repo_root`)도 반복 해석한다.
    순환 참조는 깊이 제한으로 Fail Fast.
    """

    def resolve_str(s: str, _depth: int = 0) -> str:
        if _depth > 20:
            raise ConfigError(f"Config 보간 순환 참조 의심: {s!r}")

        def repl(m: re.Match) -> str:
            return str(_lookup(config, m.group(1).strip()))

        new = _INTERP_RE.sub(repl, s)
        if new != s and _INTERP_RE.search(new):
            return resolve_str(new, _depth + 1)
        return new

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and "${" in node:
            return resolve_str(node)
        return node

    return walk(config)


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 로드해서 dict로 반환.

    `${a.b.c}` 보간(configs/default.yaml 의 paths 섹션)을 해석한 뒤 반환한다.

    Args:
        path: YAML 파일 경로.

    Returns:
        파싱 + 보간 해석된 설정 dict.

    Raises:
        ConfigError: 파일이 없거나 YAML 파싱/보간 해석 실패 시.
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

    return _resolve_interpolations(config)


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
