"""GOLD JSONL 스키마 — 발화 단위(Sample) + 코퍼스 단위(CorpusMeta).

합의 명세는 docs/data/schema.md 참조. 본 모듈은 그 명세를 코드로 옮긴 것.

사용 예:

    from project.data import load_samples, validate_samples

    samples = load_samples("BENCHMARK/GOLD/test/KsponSpeech_test_general_clean.jsonl")
    validate_samples(samples, for_test=True)   # 벤치마크 검증 — 화자 누수는 별도

핵심 원칙:
- pydantic 같은 외부 의존 없음. 표준 라이브러리만.
- frozen dataclass — 불변. (CLAUDE.md 코딩 원칙 준수)
- 에러 메시지에 "어느 줄, 어느 필드, 무엇이 잘못됐는지" 명시.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
_GENDER_VALUES = {"male", "female", "unknown"}
_LABELING_VALUES = {"human-review", "auto-stt", "raw"}

# 발화 길이 허용 범위 (초)
_DUR_MIN, _DUR_MAX = 0.1, 60.0


# ---------------------------------------------------------------------------
# Sample — 발화 단위 (GOLD JSONL 한 줄)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    """GOLD JSONL 한 줄 = 한 발화.

    필수 7개 + 선택 3개. 자세한 의미는 docs/data/schema.md §2 참조.
    """

    # ── 필수 (7개) ─────────────────────────────────────────────
    key: str                    # 전 데이터셋 고유 ID
    audio: str                  # JSONL 기준 상대경로 (load_samples 가 resolve)
    duration_sec: float         # 16kHz 기준 실 길이
    text: str                   # 원시 전사
    text_normalized: str        # 정규화 거친 텍스트 (평가 비교용)
    corpus_id: str              # 코퍼스 메타 lookup 키
    speaker_id: str             # 화자 ID — 누수 검증에 필수

    # ── 선택 (3개) — test 셋에선 강력 권장 ────────────────────
    age_group: str = "unknown"  # 예: "30대", "70대", "80대+"
    gender: str = "unknown"     # male / female / unknown
    labeling: str = "raw"       # human-review / auto-stt / raw

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
# CorpusMeta — 코퍼스 단위 (50줄 정도)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusMeta:
    """코퍼스 단위 메타. 메타시트(RAW_통합) 한 행에 해당.

    schema.md §3 참조.
    """

    corpus_id: str               # FK 키 (메타시트 Title (U))
    title: str                   # 사람 읽는 이름
    source: str = "unknown"      # AIHub / HuggingFace / 자체 등
    domain_hint: str = "unknown" # general / command / medical / dialog / ...
    style: str = "unknown"       # 낭독 / 대화-자연발화 / 대화-격식 / 독백
    language: str = "ko"
    recording_env: str = "unknown"
    license: str = "unknown"
    version: str = "unknown"
    url: str = ""
    speaker_note: str = ""
    collected_at: str = ""

    # 추가 필드 (코퍼스마다 다름) 보존
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusMeta":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        return d


# ---------------------------------------------------------------------------
# 단일 Sample 검증
# ---------------------------------------------------------------------------
def _validate_one(d: dict[str, Any], *, line_no: int, for_test: bool) -> Sample:
    """한 발화 dict 를 검증 + Sample 로 변환. 실패 시 SchemaValidationError."""

    # 필수 필드 누락
    required = ("key", "audio", "duration_sec", "text",
                "text_normalized", "corpus_id", "speaker_id")
    missing = [k for k in required if k not in d or d[k] in (None, "")]
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
    dur = float(d["duration_sec"])
    if not (_DUR_MIN <= dur <= _DUR_MAX):
        raise SchemaValidationError(
            f"[line {line_no}] duration_sec 범위 밖 ({_DUR_MIN} ~ {_DUR_MAX}): {dur}"
        )

    # 텍스트 길이
    if not str(d["text"]).strip():
        raise SchemaValidationError(f"[line {line_no}] text 가 공백/빈 문자열")
    if not str(d["text_normalized"]).strip():
        raise SchemaValidationError(f"[line {line_no}] text_normalized 가 공백/빈 문자열")

    # 선택 필드 값 검증
    gender = d.get("gender", "unknown")
    if gender not in _GENDER_VALUES:
        raise SchemaValidationError(
            f"[line {line_no}] gender 값 잘못됨 ({_GENDER_VALUES}): {gender!r}"
        )

    labeling = d.get("labeling", "raw")
    if labeling not in _LABELING_VALUES:
        raise SchemaValidationError(
            f"[line {line_no}] labeling 값 잘못됨 ({_LABELING_VALUES}): {labeling!r}"
        )

    # test 셋(벤치마크) 추가 제약
    if for_test:
        if labeling != "human-review":
            raise SchemaValidationError(
                f"[line {line_no}] 벤치마크(test) 는 labeling='human-review' 필수: {labeling!r}"
            )

    return Sample.from_dict(d)


# ---------------------------------------------------------------------------
# 배치 검증
# ---------------------------------------------------------------------------
def validate_samples(
    samples: Iterable[Sample | dict[str, Any]],
    *,
    for_test: bool = False,
    train_speakers: set[str] | None = None,
) -> list[Sample]:
    """샘플 컬렉션 일괄 검증.

    Args:
        samples: Sample 객체 또는 dict 들의 iterable.
        for_test: True 면 벤치마크(test) 추가 제약 적용
                  (labeling == 'human-review').
        train_speakers: 주어지면 *화자 누수* 검증.
                        교집합이 있으면 SchemaValidationError.

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
        out.append(_validate_one(d, line_no=i, for_test=for_test))

    # 화자 누수 검증
    if for_test and train_speakers is not None:
        test_speakers = {s.speaker_id for s in out}
        overlap = train_speakers & test_speakers
        if overlap:
            sample = sorted(overlap)[:5]
            raise SchemaValidationError(
                f"화자 누수 감지: train 과 test 가 {len(overlap)}명 겹침. "
                f"예: {sample}"
            )

    return out


# ---------------------------------------------------------------------------
# JSONL 입출력
# ---------------------------------------------------------------------------
def load_samples(path: str | Path, *, validate: bool = True,
                 for_test: bool | None = None) -> list[Sample]:
    """JSONL 파일 → list[Sample].

    Args:
        path: JSONL 경로.
        validate: True 면 검증 (기본).
        for_test: None 이면 경로에 "/test/" 가 있는지로 자동 판단.

    Returns:
        Sample 리스트.
    """

    p = Path(path).resolve()
    jsonl_dir = p.parent
    if for_test is None:
        for_test = "/test/" in str(p).replace("\\", "/")

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
    return validate_samples(raw, for_test=for_test)


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


# ---------------------------------------------------------------------------
# 코퍼스 메타 로드 (JSONL 만 지원. xlsx → JSONL 변환은 별도 스크립트 예정)
# ---------------------------------------------------------------------------
def load_corpus_meta(path: str | Path) -> dict[str, CorpusMeta]:
    """corpora.jsonl 로드 → {corpus_id: CorpusMeta}.

    xlsx 직접 로드는 별도 스크립트 (scripts/build_corpus_meta.py 예정).
    """

    p = Path(path)
    out: dict[str, CorpusMeta] = {}
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaValidationError(
                    f"[{p}:{i}] JSON 파싱 실패: {e.msg}"
                ) from e
            cid = d.get("corpus_id")
            if not cid:
                raise SchemaValidationError(
                    f"[{p}:{i}] corpus_id 누락: {d}"
                )
            if cid in out:
                raise SchemaValidationError(
                    f"[{p}:{i}] corpus_id 중복: {cid!r}"
                )
            out[cid] = CorpusMeta.from_dict(d)
    return out
