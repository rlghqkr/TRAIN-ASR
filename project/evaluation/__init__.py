"""project.evaluation — 한국어 ASR 평가 (CER 중심).

주요 진입점:
- `normalize_korean_asr(text)` — 정규화 (학습/평가 공통)
- `compute_cer(refs, hyps)` — corpus-level CER/sCER/WER
- `slice_cer(samples, slice_field=...)` — 메타 필드별 분해
- `evaluate_on_benchmark_suite(...)` — 한 모델 × 전체 Benchmark Set
- `write_text_report` / `write_json_report` / `write_diff_file` — 리포트 산출

참고 양식: practice/2601_sensevoice_train/results/<run>/
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .cer import CerResult, compute_cer, slice_cer, bootstrap_cer_ci
from .normalize import normalize_korean_asr, to_jamo
from .report import write_diff_file, write_json_report, write_text_report

__all__ = [
    "normalize_korean_asr",
    "to_jamo",
    "compute_cer",
    "slice_cer",
    "bootstrap_cer_ci",
    "CerResult",
    "write_text_report",
    "write_json_report",
    "write_diff_file",
    "evaluate_on_benchmark_suite",
]


def evaluate_on_benchmark_suite(
    *,
    model_name: str,
    predict_fn: Callable[[list[str]], list[str]],
    benchmark_paths: dict[str, str | Path],
    out_dir: str | Path,
    slice_fields: tuple[str, ...] = ("age_group", "gender", "corpus_id"),
    batch_size: int = 16,
    save_diff: bool = True,
) -> dict[str, CerResult]:
    """한 모델 × 여러 벤치마크 평가 → 리포트 일괄 생성.

    Args:
        model_name: 결과 폴더/리포트에 들어갈 모델 이름.
        predict_fn: 오디오 경로 리스트 → 예측 텍스트 리스트.
                    모델별 어댑터에서 만들어 전달.
        benchmark_paths: {benchmark_id: 'BENCHMARK/GOLD/test/<bench>.jsonl'}
        out_dir: 출력 폴더. 보통 'BENCHMARK/results/<exp>/' 또는 outputs/.../eval/
        slice_fields: 슬라이스 분해할 메타 필드.
        batch_size: predict_fn 호출 시 배치 크기.
        save_diff: True 면 <run>_diff.txt 생성.

    Returns:
        {benchmark_id: CerResult}
    """
    from project.data import load_samples
    import math

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_benchmark: dict[str, CerResult] = {}
    per_benchmark_slice: dict[str, dict[str, dict[str, CerResult]]] = {}
    all_pred_samples: list[dict] = []

    for bench_id, bench_path in benchmark_paths.items():
        samples = load_samples(bench_path)
        audios = [s.audio for s in samples]
        refs = [s.text_normalized for s in samples]

        # 배치 추론
        preds: list[str] = []
        for i in range(0, len(audios), batch_size):
            batch = audios[i:i + batch_size]
            preds.extend(predict_fn(batch))

        # 정규화 (모델 출력은 raw — normalize 거치고 비교)
        preds_norm = [normalize_korean_asr(p) for p in preds]

        result = compute_cer(refs, preds_norm, normalize=False)
        per_benchmark[bench_id] = result

        # 슬라이스
        sample_dicts: list[dict] = []
        for s, pred_raw, pred_norm in zip(samples, preds, preds_norm):
            d = s.to_dict()
            d["prediction_raw"] = pred_raw
            d["prediction_normalized"] = pred_norm
            d["benchmark_id"] = bench_id
            sample_dicts.append(d)

        per_benchmark_slice[bench_id] = {
            f: slice_cer(sample_dicts, slice_field=f) for f in slice_fields
        }

        all_pred_samples.extend(sample_dicts)

        # 벤치마크별 디테일 JSON (분석용)
        bench_out = out_dir / bench_id
        bench_out.mkdir(exist_ok=True)
        (bench_out / "predictions.jsonl").write_text(
            "\n".join(__import__("json").dumps(d, ensure_ascii=False)
                      for d in sample_dicts),
            encoding="utf-8",
        )

    # ── 종합 리포트 ──────────────────────────────────────
    write_json_report(
        model_name, per_benchmark,
        out_dir / "evaluation_report.json",
        slice_results=per_benchmark_slice,
    )
    write_text_report(
        model_name, per_benchmark,
        out_dir / "evaluation_report.txt",
        slice_results=per_benchmark_slice,
    )
    if save_diff:
        write_diff_file(
            model_name, all_pred_samples,
            out_dir / f"{model_name}_diff.txt",
        )

    return per_benchmark
