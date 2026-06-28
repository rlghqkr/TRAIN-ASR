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
    slice_fields: tuple[str, ...] = ("age", "gender"),
    batch_size: int = 16,
    save_diff: bool = True,
    sample_frac: float = 1.0,
    sample_seed: int = 42,
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
        sample_frac: 벤치마크별로 평가에 쓸 샘플 비율. 1.0=전량, 0.1=10%만.
                     빠른 확인용. (0, 1] 범위. 시간 단축 ↔ CER 변동성 trade-off.
        sample_seed: 부분 샘플링(sample_frac<1) 시 재현성용 시드. 같은 시드면 매번 같은 부분집합.

    Returns:
        {benchmark_id: CerResult}
    """
    from project.data import load_samples
    import random

    # Fail Fast: 비율은 (0, 1] 범위만 허용
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(
            f"sample_frac 은 (0, 1] 범위여야 합니다 (1.0=전량, 0.1=10%). 받은 값: {sample_frac}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_benchmark: dict[str, CerResult] = {}
    per_benchmark_slice: dict[str, dict[str, dict[str, CerResult]]] = {}
    all_pred_samples: list[dict] = []

    from tqdm.auto import tqdm

    n_bench = len(benchmark_paths)
    for bi, (bench_id, bench_path) in enumerate(benchmark_paths.items(), 1):
        # 평가는 벤치마크 데이터 몇 줄이 스키마 위반이어도 죽지 않게 건너뛴다(경고 로그 남김).
        # 학습 경로(load_samples 기본값)는 그대로 Fail-Fast 유지.
        samples = load_samples(bench_path, skip_invalid=True)

        # 부분 샘플링: sample_frac<1 이면 시드 고정 무작위 부분집합만 평가.
        # 정렬된 인덱스로 뽑아 원본 순서 유지 → 같은 시드면 항상 같은 부분집합(재현성).
        if sample_frac < 1.0:
            n_total = len(samples)
            n_keep = max(1, round(n_total * sample_frac))
            idx = sorted(random.Random(sample_seed).sample(range(n_total), n_keep))
            samples = [samples[i] for i in idx]

        audios = [s.audio for s in samples]
        refs = [s.text_norm for s in samples]

        # 배치 추론 (진행바: 벤치마크별 배치 진행)
        preds: list[str] = []
        n_batches = (len(audios) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(audios), batch_size),
            total=n_batches,
            desc=f"[{bi}/{n_bench}] {bench_id} ({len(audios)} utts)",
            unit="batch",
        ):
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
