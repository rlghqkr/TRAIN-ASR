"""기본 sanity check 테스트."""

from pathlib import Path

import pytest


def test_truth() -> None:
    assert True


def test_seed_utility() -> None:
    """seed_everything 임포트 가능 여부."""
    from project.utils import seed_everything

    # 임의 시드 적용 — 에러 없이 통과해야 함
    seed_everything(42)


def test_config_loader(tmp_path: Path) -> None:
    """YAML config 로더 동작 확인."""
    from project.utils import load_config
    from project.utils.config import ConfigError, get_required

    config_file = tmp_path / "test.yaml"
    config_file.write_text("foo: 1\nbar:\n  baz: 2\n", encoding="utf-8")

    cfg = load_config(config_file)
    assert cfg["foo"] == 1
    assert get_required(cfg, "bar.baz") == 2

    with pytest.raises(ConfigError):
        get_required(cfg, "missing.key")
