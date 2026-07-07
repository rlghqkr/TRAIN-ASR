"""스모크 테스트용 소량 학습셋 만들기.

원본 SILVER transcript.jsonl 을 **읽기만** 하고, 앞부분 N 개 발화를 떼어
train / val (서로 겹치지 않게) 로 분리해 data/SILVER/ 에 새 파일로 저장한다.
audio 경로는 원본 JSONL 위치 기준으로 **절대경로로 resolve 해 기록** —
원본 오디오는 그대로 둔 채 학습 시 경로가 깨지지 않게 한다.

기본값은 configs/smoke.yaml 규격(train 24 / val 6).

사용법:
    # 기본 (Zeroth SILVER → train 24 / val 6)
    python scripts/prep_smoke_data.py

    # 원본/개수/출력 위치 바꾸기
    python scripts/prep_smoke_data.py \
        --src /data/ASR/BENCHMARK/SILVER/OpenSLR_SLR40_Zeroth_ko/transcript.jsonl \
        --out-dir data/SILVER --n-train 24 --n-val 6
"""

import argparse
import json
import sys
from pathlib import Path

import structlog


# 프로젝트 루트를 import 경로에 추가 (editable install 안 했을 때 대비)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.data import save_samples, validate_samples  # noqa: E402


logger = structlog.get_logger()

DEFAULT_SRC = "/data/ASR/BENCHMARK/SILVER/OpenSLR_SLR40_Zeroth_ko/transcript.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="스모크용 소량 학습셋 분리")
    p.add_argument("--src", default=DEFAULT_SRC,
                   help=f"원본 SILVER transcript.jsonl (읽기 전용). 기본 {DEFAULT_SRC}")
    p.add_argument("--out-dir", default="data/SILVER",
                   help="train.jsonl / val.jsonl 저장 위치. 기본 data/SILVER")
    p.add_argument("--n-train", type=int, default=24, help="train 발화 수 (기본 24)")
    p.add_argument("--n-val", type=int, default=6, help="val 발화 수 (기본 6)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log = logger.bind(src=args.src, n_train=args.n_train, n_val=args.n_val)

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"원본 transcript.jsonl 없음: {src}")

    n_need = args.n_train + args.n_val
    src_dir = src.resolve().parent

    # 앞부분 n_need 줄만 읽는다 (원본은 수정·이동 없이 읽기만).
    raw: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))
            if len(raw) >= n_need:
                break

    if len(raw) < n_need:
        raise ValueError(
            f"원본 발화가 부족합니다: 필요 {n_need}, 실제 {len(raw)} ({src})"
        )

    # audio 상대경로 → 원본 JSONL 위치 기준 절대경로로 resolve (load_samples 와 동일 규칙).
    # 원본 오디오는 그대로 두고, JSONL 만 실제 wav 위치를 절대경로로 가리키게 한다.
    for d in raw:
        audio = str(d.get("audio", ""))
        if audio and not Path(audio).is_absolute():
            d["audio"] = str((src_dir / audio).resolve())

    # 학습/평가 공통 스키마 검증 (Fail Fast).
    samples = validate_samples(raw)

    # 절대경로가 실제 파일을 가리키는지 미리 확인 (경로 깨짐 조기 발견).
    missing = [s.audio for s in samples if not Path(s.audio).exists()]
    if missing:
        raise FileNotFoundError(
            f"audio 파일을 찾을 수 없음 ({len(missing)}건). 예: {missing[0]}"
        )

    train = samples[:args.n_train]
    val = samples[args.n_train:n_need]   # train 과 겹치지 않는 다음 구간

    out_dir = Path(args.out_dir)
    n_tr = save_samples(train, out_dir / "train.jsonl")
    n_va = save_samples(val, out_dir / "val.jsonl")

    log.info(
        "스모크 학습셋 생성 완료",
        train_jsonl=str(out_dir / "train.jsonl"),
        val_jsonl=str(out_dir / "val.jsonl"),
        n_train=n_tr,
        n_val=n_va,
    )
    print(f"✅ train {n_tr} / val {n_va} → {out_dir}/  (audio 절대경로, 원본 무수정)")


if __name__ == "__main__":
    main()
