"""평가 리포트 생성 — 참고 프로젝트(`practice/2601_sensevoice_train/results/`) 양식.

산출물:
- evaluation_report.txt   : 사람 읽는 요약 (Markdown 풍 텍스트)
- evaluation_report.json  : 기계 읽기용
- <run>_diff.txt          : 오답 분석 (틀린 발화 + ref / hyp)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .cer import CerResult


# ───────────────────────────────────────────────────────────────────────
# JSON 리포트
# ───────────────────────────────────────────────────────────────────────
def write_json_report(
    model_name: str,
    per_benchmark: dict[str, CerResult],
    out_path: str | Path,
    *,
    slice_results: dict[str, dict[str, CerResult]] | None = None,
    meta: dict | None = None,
) -> Path:
    """기계 읽기용 JSON 리포트.

    구조:
        {
          "model": "<model_name>",
          "date": "...",
          "meta": {...},
          "benchmarks": {
            "<bench_id>": {"cer": ..., "scer": ..., "wer": ..., "samples": N},
            ...
          },
          "slices": {
            "<bench_id>": {
              "<field>": {"<value>": {...}, ...}
            }
          }
        }
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = {
        "model": model_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "benchmarks": {
            bid: _cer_to_dict(r) for bid, r in per_benchmark.items()
        },
    }
    if slice_results:
        body["slices"] = {
            bid: {
                f: {v: _cer_to_dict(r) for v, r in sl.items()}
                for f, sl in fields.items()
            }
            for bid, fields in slice_results.items()
        }

    out_path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def _cer_to_dict(r: CerResult) -> dict:
    d = asdict(r)
    # per_sample_cer 은 JSON 크기 줄이려 따로 처리
    d.pop("per_sample_cer", None)
    return d


# ───────────────────────────────────────────────────────────────────────
# 텍스트 리포트 (사람 읽기용)
# ───────────────────────────────────────────────────────────────────────
def write_text_report(
    model_name: str,
    per_benchmark: dict[str, CerResult],
    out_path: str | Path,
    *,
    slice_results: dict[str, dict[str, CerResult]] | None = None,
) -> Path:
    """사람 읽는 텍스트 리포트.

    양식: 참고 프로젝트의 evaluation_report.txt 와 일관.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    add = lines.append

    add("=" * 80)
    add(f"📊 ASR Evaluation Report — {model_name}")
    add(f"   Date: {datetime.now().isoformat(timespec='seconds')}")
    add("=" * 80)
    add("")

    # ── §1 Benchmark Set 종합 ─────────────────────────────────────
    add("## 1. Benchmark Set Results (한국어 CER 표준)")
    add("-" * 80)
    add(f"{'Benchmark':<55} {'CER (%)':>10} {'sCER (%)':>10} {'Samples':>10}")
    add("-" * 80)

    total_samples = 0
    weighted_cer = 0.0
    for bid, r in per_benchmark.items():
        add(f"{bid:<55} {r.cer:>10.2f} {r.scer:>10.2f} {r.samples:>10,}")
        total_samples += r.samples
        weighted_cer += r.cer * r.samples
    add("-" * 80)
    if total_samples > 0:
        add(f"{'Weighted Average':<55} {weighted_cer / total_samples:>10.2f} {'':>10} {total_samples:>10,}")
    add("")

    # ── §2 슬라이스 ─────────────────────────────────────
    if slice_results:
        add("## 2. Slice Analysis (메타 필드별)")
        add("-" * 80)
        for bid, fields in slice_results.items():
            add(f"\n### {bid}")
            for fname, slices in fields.items():
                add(f"  [by {fname}]")
                add(f"  {'value':<20} {'CER (%)':>10} {'samples':>10}")
                # 약한 슬라이스가 위로
                items = sorted(slices.items(), key=lambda kv: -kv[1].cer)
                for v, r in items:
                    add(f"  {v:<20} {r.cer:>10.2f} {r.samples:>10,}")
        add("")

    # ── §3 약체 슬라이스 강조 ──────────────────────────
    if slice_results:
        add("## 3. Worst Slices (CER 높은 순)")
        add("-" * 80)
        all_slices: list[tuple[str, str, str, CerResult]] = []
        for bid, fields in slice_results.items():
            for fname, slices in fields.items():
                for v, r in slices.items():
                    if r.samples >= 30:   # 통계적 의미
                        all_slices.append((bid, fname, v, r))
        all_slices.sort(key=lambda x: -x[3].cer)
        for bid, fname, v, r in all_slices[:10]:
            add(f"  {bid} / {fname}={v:<14} CER {r.cer:>6.2f}%  (n={r.samples})")
        add("")

    add("=" * 80)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ───────────────────────────────────────────────────────────────────────
# Diff 파일 (오답 분석)
# ───────────────────────────────────────────────────────────────────────
def write_diff_file(
    run_name: str,
    samples: Sequence[dict],
    out_path: str | Path,
    *,
    ref_field: str = "text_norm",
    hyp_field: str = "prediction_normalized",
    max_samples: int | None = None,
) -> Path:
    """틀린 발화만 모은 diff 파일.

    양식: 참고 프로젝트의 <run>_diff.txt 와 일관.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diffs = [
        s for s in samples
        if str(s.get(ref_field, "")) != str(s.get(hyp_field, ""))
    ]
    if max_samples is not None:
        diffs = diffs[:max_samples]

    lines: list[str] = []
    add = lines.append
    add(f"# ASR Differences ({run_name})")
    add(f"# Total mismatches: {len(diffs)}")
    add(f"# Date: {datetime.now().isoformat(timespec='seconds')}")
    add("")
    add("=" * 80)

    for i, s in enumerate(diffs, 1):
        add(f"\n[Sample {i}] key={s.get('key', '?')}")
        if s.get("audio"):
            add(f"Audio:    {s['audio']}")
        add(f"Text REF: {s.get(ref_field, '')}")
        add(f"Text HYP: {s.get(hyp_field, '')} ✗")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
