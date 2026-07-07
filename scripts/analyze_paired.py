"""발화 단위 paired 분석 — baseline vs 도메인 모델 (PPT 자료 생성용).

두 모델의 predictions.jsonl(같은 벤치마크)을 발화 key 로 짝지어:
  ① win/tie/lose 집계          — "평균의 함정" 점검 (발화별 개선/동일/회귀)
  ② paired bootstrap 95% CI    — corpus CER 차이의 신뢰구간 (유의성)
  ③ S/D/I 오류 분해            — 치환/삭제/삽입 비율 변화 (오류 유형 EDA)
  ④ 숫자 포함 발화 슬라이스     — 숫자 오류 개선 여부
  ⑤ 정성 예시 후보             — 최대 개선/회귀 발화 top-N (ref/base/domain 나란히)

사용:
    python scripts/analyze_paired.py \
        --base   BENCHMARK/results/whisper_baseline_frac10__260704_013425 \
        --domain BENCHMARK/results/whisper_phone8k_frac10__260704_182214 \
        --bench  AIHub_LowQualityPhoneVoice \
        --out    reports/paired_LowQualityPhone.md

    # 여러 벤치마크 요약만 (win/lose + CI):
    python scripts/analyze_paired.py --base ... --domain ... --all

산출:
    --out 마크다운 리포트 + 같은 이름의 .csv (발화별 paired CER — 직접 EDA 용)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _require_jiwer():
    try:
        import jiwer
        return jiwer
    except ImportError as e:
        raise ImportError("jiwer 필요: pip install jiwer") from e


def load_predictions(result_dir: Path, bench_id: str) -> dict[str, dict]:
    """predictions.jsonl → {key: row}. ref 는 text_norm(평가 시 정규화본)."""
    p = result_dir / bench_id / "predictions.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"predictions 없음: {p}")
    out: dict[str, dict] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["key"]] = d
    return out


def utt_cer(jiwer, ref: str, hyp: str) -> float | None:
    """발화 CER (%). 빈 ref 는 None (제외)."""
    if not ref.strip():
        return None
    try:
        return jiwer.cer(ref, hyp) * 100
    except Exception:
        return None


def corpus_cer(jiwer, refs: list[str], hyps: list[str]) -> float:
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return 0.0
    r, h = zip(*pairs)
    return jiwer.cer(list(r), list(h)) * 100


def sdi_counts(jiwer, refs: list[str], hyps: list[str]) -> dict[str, int]:
    """corpus 수준 치환/삭제/삽입/정답 문자 수."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    r, h = zip(*pairs)
    o = jiwer.process_characters(list(r), list(h))
    return {"sub": o.substitutions, "del": o.deletions,
            "ins": o.insertions, "hit": o.hits}


def paired_bootstrap_ci(
    jiwer, refs: list[str], base_hyps: list[str], dom_hyps: list[str],
    *, n_iter: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    """corpus CER 차이(domain - base)의 95% CI. 발화 단위 재표본(paired).

    Returns: (diff, lo, hi) — % 포인트. CI 가 0 을 안 걸치면 유의.
    """
    rng = random.Random(seed)
    n = len(refs)
    diff = corpus_cer(jiwer, refs, dom_hyps) - corpus_cer(jiwer, refs, base_hyps)
    diffs: list[float] = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        rs = [refs[i] for i in idx]
        bs = [base_hyps[i] for i in idx]
        ds = [dom_hyps[i] for i in idx]
        diffs.append(corpus_cer(jiwer, rs, ds) - corpus_cer(jiwer, rs, bs))
    diffs.sort()
    lo = diffs[int(n_iter * 0.025)]
    hi = diffs[int(n_iter * 0.975)]
    return diff, lo, hi


_DIGIT_RE = re.compile(r"[0-9]")


def analyze_bench(jiwer, base_dir: Path, dom_dir: Path, bench_id: str,
                  *, top_n: int = 10) -> dict:
    """한 벤치마크 paired 분석. 결과 dict 반환 (리포트 작성용)."""
    base = load_predictions(base_dir, bench_id)
    dom = load_predictions(dom_dir, bench_id)
    keys = sorted(set(base) & set(dom))
    if not keys:
        raise ValueError(f"{bench_id}: 공통 발화 없음 (같은 벤치로 평가했는지 확인)")

    rows: list[dict] = []
    for k in keys:
        ref = base[k]["text_norm"]
        bh = base[k]["prediction_normalized"]
        dh = dom[k]["prediction_normalized"]
        bc = utt_cer(jiwer, ref, bh)
        dc = utt_cer(jiwer, ref, dh)
        if bc is None or dc is None:
            continue
        rows.append({"key": k, "ref": ref, "base_hyp": bh, "dom_hyp": dh,
                     "base_cer": bc, "dom_cer": dc, "delta": dc - bc,
                     "has_digit": bool(_DIGIT_RE.search(ref))})

    win = sum(1 for r in rows if r["delta"] < -1e-9)
    lose = sum(1 for r in rows if r["delta"] > 1e-9)
    tie = len(rows) - win - lose

    refs = [r["ref"] for r in rows]
    bhs = [r["base_hyp"] for r in rows]
    dhs = [r["dom_hyp"] for r in rows]

    diff, lo, hi = paired_bootstrap_ci(jiwer, refs, bhs, dhs)

    base_sdi = sdi_counts(jiwer, refs, bhs)
    dom_sdi = sdi_counts(jiwer, refs, dhs)

    # 숫자 포함 발화 슬라이스
    dig = [r for r in rows if r["has_digit"]]
    digit_slice = None
    if dig:
        digit_slice = {
            "n": len(dig),
            "base": corpus_cer(jiwer, [r["ref"] for r in dig], [r["base_hyp"] for r in dig]),
            "dom": corpus_cer(jiwer, [r["ref"] for r in dig], [r["dom_hyp"] for r in dig]),
        }

    by_delta = sorted(rows, key=lambda r: r["delta"])
    return {
        "bench": bench_id, "n": len(rows),
        "base_cer": corpus_cer(jiwer, refs, bhs),
        "dom_cer": corpus_cer(jiwer, refs, dhs),
        "win": win, "tie": tie, "lose": lose,
        "diff": diff, "ci": (lo, hi),
        "base_sdi": base_sdi, "dom_sdi": dom_sdi,
        "digit": digit_slice,
        "improved": by_delta[:top_n],          # 최대 개선
        "regressed": by_delta[-top_n:][::-1],  # 최대 회귀
        "rows": rows,
    }


def write_report(res: dict, out_path: Path) -> None:
    L: list[str] = []
    b = res
    L.append(f"# Paired 분석 — {b['bench']} (n={b['n']:,})\n")
    L.append(f"## Corpus CER\n- baseline: **{b['base_cer']:.2f}%**  |  domain: **{b['dom_cer']:.2f}%**  |  Δ = **{b['diff']:+.2f}%p**\n")

    lo, hi = b["ci"]
    sig = "✅ 유의 (CI가 0을 포함하지 않음)" if hi < 0 or lo > 0 else "⚠️ 비유의 (CI가 0 포함)"
    L.append(f"## 통계적 유의성 (paired bootstrap, 1000회, seed 42)\n- ΔCER 95% CI: **[{lo:+.2f}, {hi:+.2f}] %p** → {sig}\n")

    n = b["n"]
    L.append("## 발화 단위 win/tie/lose (평균의 함정 점검)\n")
    L.append(f"| 판정 | 발화 수 | 비율 |\n|---|---|---|")
    L.append(f"| 개선 (win) | {b['win']:,} | {100*b['win']/n:.1f}% |")
    L.append(f"| 동일 (tie) | {b['tie']:,} | {100*b['tie']/n:.1f}% |")
    L.append(f"| 회귀 (lose) | {b['lose']:,} | {100*b['lose']/n:.1f}% |\n")

    L.append("## 오류 유형 분해 (문자 단위 S/D/I)\n")
    L.append("| 모델 | 치환(S) | 삭제(D) | 삽입(I) |\n|---|---|---|---|")
    for name, s in (("baseline", b["base_sdi"]), ("domain", b["dom_sdi"])):
        tot = s["sub"] + s["del"] + s["hit"]
        L.append(f"| {name} | {s['sub']:,} ({100*s['sub']/tot:.1f}%) | {s['del']:,} ({100*s['del']/tot:.1f}%) | {s['ins']:,} |")
    L.append("")

    if b["digit"]:
        d = b["digit"]
        L.append(f"## 숫자 포함 발화 (n={d['n']:,})\n- baseline {d['base']:.2f}% → domain {d['dom']:.2f}%  (Δ {d['dom']-d['base']:+.2f}%p)\n")

    L.append("## 정성 예시 후보 — 최대 개선 (PPT 용)\n")
    for r in b["improved"]:
        L.append(f"- `[{r['key']}]` base {r['base_cer']:.0f}% → dom {r['dom_cer']:.0f}%")
        L.append(f"  - ref : {r['ref']}")
        L.append(f"  - base: {r['base_hyp']}")
        L.append(f"  - dom : {r['dom_hyp']}")
    L.append("\n## 최대 회귀 (정직한 분석용)\n")
    for r in b["regressed"]:
        L.append(f"- `[{r['key']}]` base {r['base_cer']:.0f}% → dom {r['dom_cer']:.0f}%")
        L.append(f"  - ref : {r['ref']}")
        L.append(f"  - base: {r['base_hyp']}")
        L.append(f"  - dom : {r['dom_hyp']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")

    # 발화별 CSV (직접 EDA 용)
    import csv
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "base_cer", "dom_cer", "delta",
                                          "has_digit", "ref", "base_hyp", "dom_hyp"])
        w.writeheader()
        for r in b["rows"]:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"[저장] {out_path}\n[저장] {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="baseline vs domain paired 분석")
    ap.add_argument("--base", required=True, help="baseline 결과 폴더 (results/<name>__<ts>)")
    ap.add_argument("--domain", required=True, help="domain 결과 폴더")
    ap.add_argument("--bench", default=None, help="벤치마크 ID (상세 리포트)")
    ap.add_argument("--all", action="store_true", help="공통 벤치마크 전체 요약(win/lose+CI)")
    ap.add_argument("--out", default=None, help="리포트 출력 경로 (.md)")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    jiwer = _require_jiwer()
    base_dir, dom_dir = Path(args.base), Path(args.domain)

    if args.all:
        benches = sorted(d.name for d in base_dir.iterdir()
                         if d.is_dir() and (d / "predictions.jsonl").exists()
                         and (dom_dir / d.name / "predictions.jsonl").exists())
        print(f"{'benchmark':35s} {'base':>7} {'dom':>7} {'Δ':>7}  {'95% CI':>18} {'win':>6} {'tie':>6} {'lose':>6}")
        print("-" * 100)
        for bid in benches:
            r = analyze_bench(jiwer, base_dir, dom_dir, bid, top_n=0)
            lo, hi = r["ci"]
            print(f"{bid:35s} {r['base_cer']:7.2f} {r['dom_cer']:7.2f} {r['diff']:+7.2f}  "
                  f"[{lo:+6.2f},{hi:+6.2f}] {r['win']:6d} {r['tie']:6d} {r['lose']:6d}")
        return

    if not args.bench:
        ap.error("--bench <ID> 또는 --all 필요")
    res = analyze_bench(jiwer, base_dir, dom_dir, args.bench, top_n=args.top_n)
    out = Path(args.out) if args.out else ROOT / "reports" / f"paired_{args.bench}.md"
    write_report(res, out)


if __name__ == "__main__":
    main()
