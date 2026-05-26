"""project.data — 데이터 스키마 / 로딩 / 어댑터.

공통 GOLD JSONL (한 줄 = 한 발화) 을 다루는 모듈.
모델별 변환은 `project.data.adapters` 하위에 둔다 (향후 추가).
"""

from .schema import (
    CorpusMeta,
    Sample,
    SchemaValidationError,
    load_corpus_meta,
    load_samples,
    save_samples,
    validate_samples,
)

__all__ = [
    "Sample",
    "CorpusMeta",
    "SchemaValidationError",
    "load_samples",
    "save_samples",
    "load_corpus_meta",
    "validate_samples",
]
