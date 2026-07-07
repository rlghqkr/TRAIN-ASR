"""도메인 데이터 균등 샘플링 (각 데이터셋 동일 개수) — 어블레이션용.

목적:
    "baseline + 각 도메인 1개씩" 비교(어블레이션)에서 데이터 '양'을 통제한다.
    각 도메인 데이터셋을 **같은 N개**로 시드 고정 샘플링 → 데이터셋 per-sample 순수
    기여도를 공정하게 비교(양 차이가 교란되지 않게).

N 기준:
    N = 237,559 = 2:1 실험에서 가장 작았던 Counseling 의 2:1 몫.
    (가장 작은 값에 맞춰 다른 둘을 그 크기로 다운샘플)

방식:
    - 각 도메인 SILVER 전량에서 N개 무작위 샘플 (시드 42, 재현 가능)
    - audio 경로가 절대경로(RAW 참조)라 출력 위치 무관

실행:
    python scripts/sample_domain_equal.py

산출물:
    /data/ASR/_train_ablat_kiho/<dataset>/transcript.jsonl  (도메인 3개, 각 N개)
"""

from __future__ import annotations

import os
import random

SEED = 42
N = 237559   # 각 도메인 데이터셋을 이 개수로 통일 (Counseling 2:1 몫 기준)

DOMAIN_NAMES = [
    "AIHub_LowQualityPhoneVoice",
    "AIHub_CounselingSpeech",
    "AIHub_WelfareCounsel",
]
SRC_TMPL = "/data/ASR/TRAIN/SILVER/{name}/transcript.jsonl"
OUT_ROOT = "/data/ASR/_train_ablat_kiho"


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    rng = random.Random(SEED)
    print(f"목표: 각 도메인 {N:,}개로 통일\n")
    for name in DOMAIN_NAMES:
        src = SRC_TMPL.format(name=name)
        n_total = count_lines(src)
        n_keep = min(N, n_total)
        idx = set(rng.sample(range(n_total), n_keep))

        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, "transcript.jsonl")
        with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
            for i, line in enumerate(f):
                if i in idx:
                    out.write(line)

        print(f"  {name}: {n_total:,} -> {n_keep:,}\n    {dst}")
    print(f"\n[완료] 각 도메인 {N:,}개 균등 샘플 생성")


if __name__ == "__main__":
    main()
