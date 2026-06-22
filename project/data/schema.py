"""발화 JSONL 스키마 — 발화 단위(Sample).

합의 명세는 docs/data/schema.md 참조. 본 모듈은 그 명세를 코드로 옮긴 것.
학습/평가가 같은 스키마를 공유한다.

사용 예:

    from project.data import load_samples, validate_samples

    samples = load_samples("BENCHMARK/data/Sample10_PracticeRef/transcript.jsonl")
    validate_samples(samples)   # 학습/평가 공통 스키마 검증

핵심 원칙:
- pydantic 같은 외부 의존 없음. 표준 라이브러리만.
- frozen dataclass — 불변. (CLAUDE.md 코딩 원칙 준수)
- 에러 메시지에 "어느 줄, 어느 필드, 무엇이 잘못됐는지" 명시.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# 예외
# ---------------------------------------------------------------------------
class SchemaValidationError(ValueError):
    """스키마 검증 실패. 메시지에 어느 줄/필드/사유 포함."""


# ---------------------------------------------------------------------------
# 허용 값 (Awesome-Korean-Speech-Recognition 컨벤션 + 사내 합의)
# ---------------------------------------------------------------------------
_GENDER_VALUES = {"F", "M", "unknown"}

# 발화 길이 허용 범위 (초)
_DUR_MIN, _DUR_MAX = 0.1, 60.0


# ---------------------------------------------------------------------------
# Sample — 발화 단위 (GOLD JSONL 한 줄)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    """발화 JSONL 한 줄 = 한 발화.

    필수 5개 + 선택 2개. 자세한 의미는 docs/data/schema.md §2 참조.
    학습/평가가 같은 스키마를 공유한다.
    """

    # ── 필수 (5개) ─────────────────────────────────────────────
    key: str                    # 전 데이터셋 고유 ID (예: "{dataset_id}__000")
    audio: str                  # JSONL 기준 상대경로 (load_samples 가 resolve)
    duration: float             # 16kHz 기준 실 길이 (초)
    text: str                   # 원시 전사 (rich)
    text_norm: str              # 정규화 거친 텍스트 (학습 타겟 + 평가 비교용)

    # ── 선택 (2개) ─────────────────────────────────────────────
    age: str = "unknown"        # 예: "30대", "70대", "10대", "6세 이하"
    gender: str = "unknown"     # F / M / unknown

    # ----- 변환 -----
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sample":
        """dict → Sample. 누락된 선택 필드는 기본값으로."""
        known = {f for f in cls.__dataclass_fields__}
        # 모르는 키는 무시 (확장 가능성 — dialect 같은 추가 필드는 어댑터에서 처리)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 단일 Sample 검증
# ---------------------------------------------------------------------------
def _validate_one(d: dict[str, Any], *, line_no: int) -> Sample:
    """한 발화 dict 를 검증 + Sample 로 변환. 실패 시 SchemaValidationError."""

    # 필수 필드 누락
    # text / text_norm 은 "" 허용 (잡음/이벤트만 있어 전사·정규화 결과가 없는 발화).
    # key / audio / duration 은 키 존재 + 빈 값/None 불가.
    required = ("key", "audio", "duration", "text", "text_norm")
    missing = [
        k for k in required
        if k not in d or d[k] is None or (k not in ("text", "text_norm") and d[k] == "")
    ]
    if missing:
        raise SchemaValidationError(
            f"[line {line_no}] 필수 필드 누락: {missing}"
        )

    # audio 경로가 비어 있는지
    if not str(d["audio"]).strip():
        raise SchemaValidationError(
            f"[line {line_no}] audio 경로가 비어 있음"
        )

    # duration 범위
    dur = float(d["duration"])
    if not (_DUR_MIN <= dur <= _DUR_MAX):
        raise SchemaValidationError(
            f"[line {line_no}] duration 범위 밖 ({_DUR_MIN} ~ {_DUR_MAX}): {dur}"
        )

    # text / text_norm 은 "" 허용 (잡음 발화 등) — 길이 검사 안 함.

    # 선택 필드 값 검증
    gender = d.get("gender", "unknown")
    if gender not in _GENDER_VALUES:
        raise SchemaValidationError(
            f"[line {line_no}] gender 값 잘못됨 ({_GENDER_VALUES}): {gender!r}"
        )

    return Sample.from_dict(d)


# ---------------------------------------------------------------------------
# 배치 검증
# ---------------------------------------------------------------------------
def validate_samples(
    samples: Iterable[Sample | dict[str, Any]],
) -> list[Sample]:
    """샘플 컬렉션 일괄 검증.

    Args:
        samples: Sample 객체 또는 dict 들의 iterable.

    Returns:
        검증 통과한 Sample 리스트.

    Raises:
        SchemaValidationError: 한 줄이라도 실패하면 즉시 (Fail Fast).
    """

    out: list[Sample] = []
    for i, s in enumerate(samples, 1):
        if isinstance(s, Sample):
            # 이미 Sample 이면 dict 로 변환해 같은 경로로 검증
            d = s.to_dict()
        else:
            d = dict(s)
        out.append(_validate_one(d, line_no=i))

    return out


# ---------------------------------------------------------------------------
# JSONL 입출력
# ---------------------------------------------------------------------------
def load_samples(path: str | Path, *, validate: bool = True) -> list[Sample]:
    """JSONL 파일 → list[Sample].

    Args:
        path: JSONL 경로.
        validate: True 면 검증 (기본).

    Returns:
        Sample 리스트.
    """

    p = Path(path).resolve()
    jsonl_dir = p.parent

    raw: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SchemaValidationError(
                    f"[{p}:{i}] JSON 파싱 실패: {e.msg}"
                ) from e

    # 상대경로 → JSONL 위치 기준 절대경로로 resolve
    for d in raw:
        audio = str(d.get("audio", ""))
        if audio and not Path(audio).is_absolute():
            d["audio"] = str((jsonl_dir / audio).resolve())

    if not validate:
        return [Sample.from_dict(d) for d in raw]
    return validate_samples(raw)


def save_samples(samples: Iterable[Sample], path: str | Path) -> int:
    """list[Sample] → JSONL. 반환값: 저장된 줄 수."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
