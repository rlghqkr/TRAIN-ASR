"""SILVER transcript.jsonl 을 직접 입력받는 평가 러너.

팀 합의(2026-06): 벤치마크 테스트는 SILVER 포맷으로 진행.
평가 파이프라인(project/data/schema.py)은 GOLD 필드명을 요구하므로
여기서 필드명을 매핑한 변환본을 만들어 전달한다. 원본 SILVER 는 읽기만 한다.

SILVER → GOLD 필드 매핑:
    duration  → duration_sec
    text_norm → text_normalized
    age       → age_group
    gender    F/M → female/male
    corpus_id(벤치마크 ID)·speaker_id(오디오 파일명)·labeling 누락 시 채움
    audio 상대경로 → SILVER 위치 기준 절대경로

NOTE: speaker_id 를 오디오 파일명으로 채우는 것은 placeholder —
화자 단위 분석/누수 검증에는 사용할 수 없음 (KsponSpeech eval 은 화자 정보 비공개).

사용:
    python scripts/eval_silver.py --config BENCHMARK/configs/eval/<exp>.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# 저장소 루트를 path 에 추가 (editable install 없을 때 대비)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from project.evaluation import evaluate_on_benchmark_suite  # noqa: E402

GENDER_MAP = {"F": "female", "M": "male", "female": "female", "male": "male"}


def convert_silver(src: Path, dst: Path, *, corpus_id: str) -> int:
    """SILVER jsonl → GOLD 필드명 jsonl. 원본 불변. 반환값: 변환된 줄 수."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            # 정답(text_normalized)이 빈 발화는 CER 평가 불가 → 제외.
            # text 에 'u/'(불명료) 같은 태그가 있어도 정규화 정답이 비면 스킵.
            if not str(d.get("text_norm") or d.get("text_normalized") or "").strip():
                skipped += 1
                continue
            if "duration" in d:
                d["duration_sec"] = d.pop("duration")
            if "text_norm" in d:
                d["text_normalized"] = d.pop("text_norm")
            if "age" in d:
                d["age_group"] = d.pop("age")
            d["gender"] = GENDER_MAP.get(d.get("gender", "unknown"), "unknown")
            d.setdefault("corpus_id", corpus_id)
            d.setdefault("speaker_id", Path(d["audio"]).stem)
            d.setdefault("labeling", "human-review")
            audio = Path(d["audio"])
            if not audio.is_absolute():
                # 변환본은 results 폴더에 저장되므로 SILVER 위치 기준으로 절대화
                d["audio"] = str((src.parent / audio).resolve())
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    if skipped:
        print(f"  [skip] 정답 전사가 빈 발화 {skipped}개 제외")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="BENCHMARK/configs/eval/*.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rec = cfg["recognizer"]

    if rec["type"] == "whisper":
        from project.data.adapters.whisper import build_predict_fn
        predict_fn = build_predict_fn(
            rec["model_path"],
            backbone=rec["backbone"],
            **rec.get("options", {}),
        )
    elif rec["type"] == "sensevoice":
        from project.data.adapters.sensevoice import build_predict_fn
        predict_fn = build_predict_fn(rec["model_path"], **rec.get("options", {}))
    else:
        raise ValueError(f"unknown recognizer.type: {rec['type']}")

    out_dir = REPO / "BENCHMARK" / "results" / rec["name"]
    conv_dir = out_dir / "_silver_converted"

    # benchmarks: {벤치마크ID: SILVER transcript.jsonl 경로}
    benchmark_paths: dict[str, Path] = {}
    for bid, silver in cfg["benchmarks"].items():
        src = Path(silver)
        if not src.exists():
            raise SystemExit(f"SILVER 없음: {src}")
        dst = conv_dir / f"{bid}.jsonl"
        n = convert_silver(src, dst, corpus_id=bid)
        print(f"[convert] {bid}: {n} samples → {dst}")
        benchmark_paths[bid] = dst

    results = evaluate_on_benchmark_suite(
        model_name=rec["name"],
        predict_fn=predict_fn,
        benchmark_paths=benchmark_paths,
        out_dir=out_dir,
        batch_size=cfg["batch_size"],
    )
    for bid, r in results.items():
        print(f"{bid}: CER {r.cer:.2f} / sCER {r.scer:.2f} / "
              f"WER {r.wer:.2f} ({r.samples} samples)")


if __name__ == "__main__":
    main()
