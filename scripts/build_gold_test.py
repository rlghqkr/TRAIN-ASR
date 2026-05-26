"""SILVER manifest → GOLD test/<benchmark>.jsonl 빌드.

사용 (예):
    python scripts/build_gold_test.py \\
        --silver data/SILVER/AIHub_FreeDialog_Senior/manifest.jsonl \\
        --benchmark-id AIHub_FreeDialog_Senior_test_elderly \\
        --filter '{"meta.age_group": ["60대","70대","80대+"]}' \\
        --sample 1500 \\
        --train data/GOLD/train.jsonl \\
        --out BENCHMARK/GOLD/test/AIHub_FreeDialog_Senior_test_elderly.jsonl

검증:
- 모든 항목 labeling == "human-review"
- text / text_normalized 비어있지 않음
- duration / 절대 경로 등 schema.py validate
- train 화자와 무중첩 (assert)

자세한 정책: docs/data/schema.md §7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# 저장소 루트를 path 에 추가 (editable install 없을 때 대비)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from project.data import (  # noqa: E402
    SchemaValidationError,
    load_samples,
    save_samples,
    validate_samples,
)
from project.data.schema import Sample  # noqa: E402
from project.evaluation import normalize_korean_asr  # noqa: E402


def _match(s_dict: dict[str, Any], filt: dict[str, Any]) -> bool:
    """간단한 dot-path 필터.

    예: filt={"meta.age_group": ["60대","70대"], "domain": "general"}
    """
    for k, v in filt.items():
        keys = k.split(".")
        cur = s_dict
        for kk in keys:
            if not isinstance(cur, dict):
                return False
            cur = cur.get(kk)
            if cur is None:
                return False
        # v 가 리스트면 in 검사
        if isinstance(v, list):
            if cur not in v:
                return False
        else:
            if cur != v:
                return False
    return True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _ensure_normalized(samples: list[Sample]) -> list[Sample]:
    """text_normalized 가 비어 있거나 일치 안 하면 다시 채움."""
    fixed: list[Sample] = []
    for s in samples:
        expected = normalize_korean_asr(s.text)
        if s.text_normalized != expected:
            d = s.to_dict()
            d["text_normalized"] = expected
            fixed.append(Sample.from_dict(d))
        else:
            fixed.append(s)
    return fixed


def _load_train_speakers(train_path: Path | None) -> set[str]:
    if not train_path:
        return set()
    if not train_path.exists():
        print(f"WARN: train manifest 없음: {train_path}  — 화자 누수 검증 스킵", file=sys.stderr)
        return set()
    speakers: set[str] = set()
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("speaker_id")
            if sid:
                speakers.add(sid)
    return speakers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silver", required=True,
                    help="SILVER manifest JSONL (또는 GOLD JSONL — 같은 스키마)")
    ap.add_argument("--benchmark-id", required=True,
                    help="<corpus>_test_<속성>_<조건> 패턴")
    ap.add_argument("--filter", default="{}",
                    help='JSON dict — 예: \'{"meta.age_group": ["60대"]}\'')
    ap.add_argument("--sample", type=int, default=0,
                    help="0 = 전체 통과. N > 0 이면 N 개 무작위 샘플링")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", default="data/GOLD/train.jsonl",
                    help="화자 누수 검증에 사용할 train manifest")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    silver_path = Path(args.silver)
    out_path = Path(args.out)
    train_path = Path(args.train) if args.train else None
    filt = json.loads(args.filter)

    if not silver_path.exists():
        raise SystemExit(f"입력 없음: {silver_path}")

    # 1) SILVER 로드 (검증 끄고 일단 읽기 — 누락 필드 채우려고)
    raw = _read_jsonl(silver_path)
    print(f"입력: {len(raw):,}")

    # 2) 필터
    if filt:
        raw = [r for r in raw if _match(r, filt)]
        print(f"필터 통과: {len(raw):,}")

    # 3) labeling == 'human-review' 만 (벤치마크 제약)
    raw = [r for r in raw if r.get("labeling") == "human-review"]
    print(f"human-review 통과: {len(raw):,}")

    if not raw:
        raise SystemExit("빌드할 발화가 0 — 필터 / 검수 정책 확인")

    # 4) 무작위 샘플링
    if args.sample > 0 and len(raw) > args.sample:
        rng = random.Random(args.seed)
        raw = rng.sample(raw, args.sample)
        print(f"샘플링 ({args.sample}): {len(raw):,}")

    # 5) Sample 변환 + 검증
    samples = [Sample.from_dict(r) for r in raw]
    samples = _ensure_normalized(samples)

    train_speakers = _load_train_speakers(train_path)
    try:
        samples = validate_samples(
            samples, for_test=True,
            train_speakers=train_speakers if train_speakers else None,
        )
    except SchemaValidationError as e:
        raise SystemExit(f"검증 실패: {e}")

    # 6) 저장
    n = save_samples(samples, out_path)
    print(f"\nwrote {n} samples → {out_path}")

    # 7) 통계 (분포)
    from collections import Counter
    print("\n=== 분포 ===")
    print(f"  화자 수:  {len({s.speaker_id for s in samples})}")
    print(f"  총 시간:  {sum(s.duration_sec for s in samples) / 3600:.2f} h")
    print(f"  코퍼스:   {Counter(s.corpus_id for s in samples).most_common()}")
    print(f"  gender:   {Counter(s.gender for s in samples).most_common()}")
    print(f"  age:      {Counter(s.age_group for s in samples).most_common()}")


if __name__ == "__main__":
    main()
