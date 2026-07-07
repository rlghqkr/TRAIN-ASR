#!/usr/bin/env python3
"""
benchmark_silver2gold.py

SILVER 벤치마크 데이터셋을 GOLD로 구축하는 스크립트.

규칙
----
- 코퍼스(데이터셋)의 발화 수가 THRESHOLD(기본 4000) "미만"이면
  폴더 전체를 GOLD로 그대로 하드카피한다. (심볼릭 링크 X, 실제 파일 복사)
- THRESHOLD "이상"이면 SAMPLE_SIZE(기본 4000)건으로 랜덤 샘플링하여
  해당 발화의 음성 파일과 transcript만 GOLD로 복사한다.
- 랜덤 시드는 스크립트 상단에서 고정한다.

디렉토리 구조 (입력 / 출력 동일)
    <ROOT>/<corpus>/audio/*.wav
    <ROOT>/<corpus>/transcript.jsonl
"""

import json
import random
import shutil
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
SILVER_ROOT = Path("/data/ASR/BENCHMARK/SILVER")
GOLD_ROOT   = Path("/data/ASR/BENCHMARK/GOLD")

THRESHOLD   = 4000     # 이 값 "미만"이면 전체 복사, "이상"이면 샘플링
SAMPLE_SIZE = 4000     # 샘플링 시 추출 건수
SEED        = 42       # 랜덤 시드 (스크립트 처음에 고정)

OVERWRITE   = False    # GOLD에 이미 해당 코퍼스가 있으면 덮어쓸지 여부
DRY_RUN     = False    # True면 실제 복사 없이 처리 계획만 출력

# transcript.jsonl 안에서 음성 파일 경로가 들어있는 필드 이름.
# None이면 자동 감지(아래 후보들 중 첫 줄에서 발견되는 첫 번째 키).
AUDIO_KEY = None
AUDIO_KEY_CANDIDATES = [
    "audio_filepath", "audio_path", "audio", "wav", "filepath", "path", "file",
]

# 시드 고정 (스크립트 처음에)
random.seed(SEED)
# ─────────────────────────────────────────────────────────────


def read_manifest(jsonl_path):
    """transcript.jsonl을 읽어 dict 리스트로 반환."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def detect_audio_key(rows):
    """음성 경로 필드 이름을 결정."""
    if AUDIO_KEY is not None:
        return AUDIO_KEY
    if not rows:
        return None
    for key in AUDIO_KEY_CANDIDATES:
        if key in rows[0]:
            return key
    return None


def resolve_audio_src(corpus_dir, audio_value):
    """manifest의 음성 경로 값을 실제 파일 경로로 해석."""
    p = Path(audio_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(corpus_dir / p)                  # corpus/<value>
        candidates.append(corpus_dir / "audio" / p.name)   # corpus/audio/<basename>
    for c in candidates:
        if c.exists():
            return c
    return None


def copy_sampled(corpus_dir, gold_dir, rows, audio_key):
    """샘플링된 rows의 음성 + transcript를 gold_dir로 복사."""
    if not DRY_RUN:
        gold_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    missing = 0
    for row in rows:
        src = resolve_audio_src(corpus_dir, row.get(audio_key, ""))
        if src is None:
            missing += 1
            continue
        # corpus_dir 기준 상대 경로 구조를 유지하여 복사 (audio/ 하위 유지)
        try:
            rel = src.relative_to(corpus_dir)
        except ValueError:
            rel = Path("audio") / src.name
        new_row = dict(row)
        new_row[audio_key] = rel.as_posix()  # GOLD 내부 상대 경로로 정리
        out_rows.append(new_row)
        if not DRY_RUN:
            dst = gold_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)   # 하드카피 (심볼릭 링크 아님)

    if not DRY_RUN:
        with open(gold_dir / "transcript.jsonl", "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(out_rows), missing


def process_corpus(corpus_dir):
    name = corpus_dir.name
    jsonl_path = corpus_dir / "transcript.jsonl"
    if not jsonl_path.exists():
        print(f"[SKIP]   {name}: transcript.jsonl 없음")
        return

    gold_dir = GOLD_ROOT / name
    if gold_dir.exists():
        if OVERWRITE:
            if not DRY_RUN:
                shutil.rmtree(gold_dir)
        else:
            print(f"[SKIP]   {name}: GOLD에 이미 존재 (OVERWRITE=False)")
            return

    rows = read_manifest(jsonl_path)
    n = len(rows)

    if n < THRESHOLD:
        # 전체 그대로 하드카피 (symlinks=False = 실제 복사)
        if not DRY_RUN:
            shutil.copytree(corpus_dir, gold_dir, symlinks=False)
        print(f"[COPY]   {name}: {n}건 < {THRESHOLD} → 전체 복사")
    else:
        audio_key = detect_audio_key(rows)
        if audio_key is None:
            print(f"[ERROR]  {name}: 음성 경로 필드를 찾을 수 없음. "
                  f"AUDIO_KEY를 직접 지정하세요. (후보: {AUDIO_KEY_CANDIDATES})")
            return
        k = min(SAMPLE_SIZE, n)
        sampled = random.sample(rows, k)
        copied, missing = copy_sampled(corpus_dir, gold_dir, sampled, audio_key)
        msg = f"[SAMPLE] {name}: {n}건 ≥ {THRESHOLD} → {copied}건 샘플 복사"
        if missing:
            msg += f"  (음성 누락 {missing}건 제외)"
        print(msg)


def main():
    if not SILVER_ROOT.exists():
        sys.exit(f"SILVER 경로가 없습니다: {SILVER_ROOT}")
    if not DRY_RUN:
        GOLD_ROOT.mkdir(parents=True, exist_ok=True)

    corpora = sorted(d for d in SILVER_ROOT.iterdir() if d.is_dir())
    if not corpora:
        sys.exit(f"코퍼스 폴더가 없습니다: {SILVER_ROOT}")

    print(f"SILVER : {SILVER_ROOT}")
    print(f"GOLD   : {GOLD_ROOT}")
    print(f"THRESHOLD={THRESHOLD}, SAMPLE_SIZE={SAMPLE_SIZE}, SEED={SEED}, "
          f"OVERWRITE={OVERWRITE}, DRY_RUN={DRY_RUN}")
    print("-" * 64)

    for corpus_dir in corpora:
        process_corpus(corpus_dir)

    print("-" * 64)
    print("완료")


if __name__ == "__main__":
    main()