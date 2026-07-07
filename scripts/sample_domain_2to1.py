"""도메인 학습 데이터 2:1 다운샘플링 (domain : general = 2 : 1).

배경:
    저음질 전화망 도메인 데이터(LowQualityPhone + Counseling + Welfare)가 약 768만 개로,
    일반(Kspon + Zeroth) 64만 개의 ~12배다. 자연 concat 하면 도메인이 학습을 압도해
    "베이스라인 + 도메인" 비교가 깨지고(일반 망각), 디스크/시간도 비현실적이 된다.
    → 도메인을 일반의 2배(2:1) 수준으로 줄여 리허설 믹스를 맞춘다.

방식:
    - 도메인 목표 수 = (일반 전체 발화 수) x RATIO
    - 도메인 3개 데이터셋에 "전체 크기 비례"로 목표를 배분 (주력 LowQualityPhone 가 가장 많음)
    - 시드 고정 → 매 실행 동일한 부분집합(재현 가능)
    - transcript.jsonl 의 audio 경로가 절대경로(RAW 참조)라, 줄을 그대로 복사하면
      출력 파일을 어디 두든 오디오가 정상 resolve 된다.

실행:
    python sample_domain_2to1.py

산출물:
    /data/ASR/_train_sample_kiho/<dataset>/transcript.jsonl  (도메인 3개)
    → 이 경로들을 configs/whisper_phone8k_adapt_v1.yaml 의 train_jsonl 도메인 항목에 넣는다.
"""

from __future__ import annotations

import os
import random

# ─── 설정 (필요 시 여기만 수정) ──────────────────────────────────────────────
SEED = 42          # 재현용 시드 (프로젝트 표준)
RATIO = 2          # domain : general 비율. 2 = 도메인을 일반의 2배로.

# 일반(앵커) — 전량 사용. 도메인 목표 수 계산의 기준.
GENERAL_PATHS = [
    "/data/ASR/TRAIN/SILVER/AIHub_KsponSpeech/transcript.jsonl",
    "/data/ASR/TRAIN/SILVER/OpenSLR_SLR40_Zeroth_ko/transcript.jsonl",
]

# 도메인 — 아래에서 비례 배분으로 샘플링.
DOMAIN_NAMES = [
    "AIHub_LowQualityPhoneVoice",   # ★ 주력 (저음질 전화)
    "AIHub_CounselingSpeech",       # ○ 보강 (상담 전화)
    "AIHub_WelfareCounsel",         # ○ 보강 (복지상담 전화)
]

SRC_TMPL = "/data/ASR/TRAIN/SILVER/{name}/transcript.jsonl"
OUT_ROOT = "/data/ASR/_train_sample_kiho"   # kiho 쓰기 가능 디스크
# ────────────────────────────────────────────────────────────────────────────


def count_lines(path: str) -> int:
    """파일의 줄 수(= 발화 수). JSON 파싱 없이 빠르게."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    # 1) 일반 전체 발화 수 → 도메인 목표 수
    n_general = sum(count_lines(p) for p in GENERAL_PATHS)
    domain_target = n_general * RATIO
    print(f"[일반] {n_general:,} 발화  →  [도메인 목표 {RATIO}:1] {domain_target:,} 발화")

    # 2) 도메인 데이터셋별 전체 크기
    sizes = {name: count_lines(SRC_TMPL.format(name=name)) for name in DOMAIN_NAMES}
    total_domain = sum(sizes.values())
    print(f"[도메인 전체] {total_domain:,} 발화\n")

    # 3) 전체 크기 비례로 배분 + 시드 고정 샘플링
    rng = random.Random(SEED)
    kept_total = 0
    for name in DOMAIN_NAMES:
        n_total = sizes[name]
        n_keep = round(domain_target * n_total / total_domain)
        n_keep = min(n_keep, n_total)   # 안전장치 (목표가 원본보다 크면 전량)

        # 뽑을 줄 번호 집합 (정렬된 인덱스로 원본 순서 보존)
        idx = set(rng.sample(range(n_total), n_keep))

        src = SRC_TMPL.format(name=name)
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, "transcript.jsonl")

        with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
            for i, line in enumerate(f):
                if i in idx:
                    out.write(line)

        kept_total += n_keep
        pct = 100.0 * n_keep / n_total
        print(f"  {name}: {n_total:,} -> {n_keep:,} ({pct:.1f}%)\n    {dst}")

    ratio = kept_total / n_general if n_general else 0.0
    print(f"\n[완료] 도메인 샘플 합계 {kept_total:,}  (일반 {n_general:,} 대비 {ratio:.2f} : 1)")


if __name__ == "__main__":
    main()
