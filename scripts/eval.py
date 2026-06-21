"""평가 진입점 (CLI).

BENCHMARK/configs/eval/<name>.yaml 의 `recognizer` 로 모델을 로드하고,
`benchmarks` 목록을 GOLD(기본)/SILVER 경로에서 찾아 평가한 뒤
BENCHMARK/results/<recognizer.name>/ 에 리포트를 쓴다.

실제 평가 로직 SoT 는 project/evaluation/evaluate_on_benchmark_suite.
이 스크립트는 "config → predict_fn + benchmark_paths 조립 → 호출" 만 담당한다.

사용법:
    # 기본 (GOLD 경로에서 평가)
    python scripts/eval.py --config BENCHMARK/configs/eval/whisper_baseline.yaml

    # SILVER 로 평가 (가끔)
    python scripts/eval.py --config ... --stage silver

    # 학습 직후 자동 평가 — 모델 경로/이름만 덮어쓰기 (train.sh 가 사용)
    python scripts/eval.py --config ... --model-path outputs/<exp> --name <exp>
"""

import argparse
import sys
from pathlib import Path

import structlog


# 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.utils import load_config  # noqa: E402


logger = structlog.get_logger()

# 벤치마크 데이터 루트. <root>/<STAGE>/<bench_id>/transcript.jsonl
DEFAULT_BENCH_ROOT = "/data/ASR/BENCHMARK"
TRANSCRIPT_NAME = "transcript.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASR 벤치마크 평가")
    p.add_argument("--config", required=True,
                   help="평가 yaml (예: BENCHMARK/configs/eval/whisper_baseline.yaml)")
    p.add_argument("--stage", choices=["gold", "silver"], default="gold",
                   help="벤치마크 단계 경로 선택. 기본 gold. (가끔 silver)")
    p.add_argument("--bench-root", default=None,
                   help=f"벤치마크 루트 직접 지정(있으면 --stage 무시). "
                        f"기본 {DEFAULT_BENCH_ROOT}/<STAGE>. "
                        f"로컬 샘플은 BENCHMARK/data 지정.")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="평가할 bench_id 목록 (yaml benchmarks 덮어쓰기)")
    p.add_argument("--model-path", default=None,
                   help="recognizer.model_path 덮어쓰기 (학습 직후 outputs/<exp> 지정용)")
    p.add_argument("--name", default=None,
                   help="recognizer.name 덮어쓰기 (results/<name> 폴더)")
    return p.parse_args()


def build_predict_fn(recognizer: dict):
    """recognizer 설정 → 어댑터별 predict_fn."""
    rtype = recognizer["type"]
    model_path = recognizer["model_path"]
    options = dict(recognizer.get("options", {}))

    if rtype == "whisper":
        from project.data.adapters.whisper import build_predict_fn as _build
        backbone = recognizer.get("backbone", model_path)
        return _build(model_path, backbone=backbone, **options)
    if rtype == "sensevoice":
        from project.data.adapters.sensevoice import build_predict_fn as _build
        return _build(model_path, **options)
    raise ValueError(f"알 수 없는 recognizer.type: {rtype!r} (whisper | sensevoice)")


def resolve_benchmark_paths(bench_ids: list[str], *, stage: str,
                            bench_root: str | None) -> dict[str, str]:
    """bench_id 목록 → {bench_id: transcript.jsonl 절대경로}. 없으면 Fail Fast."""
    if bench_root is not None:
        root = Path(bench_root)
    else:
        root = Path(DEFAULT_BENCH_ROOT) / stage.upper()

    paths: dict[str, str] = {}
    for bid in bench_ids:
        p = root / bid / TRANSCRIPT_NAME
        if not p.exists():
            raise FileNotFoundError(
                f"벤치마크 파일 없음: {p}\n"
                f"  - --stage 를 바꾸거나(gold↔silver) --bench-root 로 경로를 지정하세요.\n"
                f"  - 로컬 샘플이면: --bench-root {ROOT / 'BENCHMARK' / 'data'}"
            )
        paths[bid] = str(p)
    return paths


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    recognizer = dict(cfg["recognizer"])
    if args.model_path:
        recognizer["model_path"] = args.model_path
    if args.name:
        recognizer["name"] = args.name

    name = recognizer["name"]
    bench_ids = args.benchmarks if args.benchmarks else cfg["benchmarks"]
    batch_size = cfg.get("batch_size", 16)

    log = logger.bind(recognizer=name, type=recognizer["type"], stage=args.stage)
    log.info("Loaded eval config", config=args.config, n_benchmarks=len(bench_ids))

    # 1) 벤치마크 경로 해석 (Fail Fast)
    benchmark_paths = resolve_benchmark_paths(
        bench_ids, stage=args.stage, bench_root=args.bench_root,
    )

    # 2) predict_fn 조립
    predict_fn = build_predict_fn(recognizer)

    # 3) 평가 실행
    from project.evaluation import evaluate_on_benchmark_suite

    out_dir = ROOT / "BENCHMARK" / "results" / name
    log.info("Evaluation started", model_path=recognizer["model_path"],
             out_dir=str(out_dir))

    results = evaluate_on_benchmark_suite(
        model_name=name,
        predict_fn=predict_fn,
        benchmark_paths=benchmark_paths,
        out_dir=out_dir,
        batch_size=batch_size,
        save_diff=True,
    )

    # 4) 요약 출력
    print("\n" + "=" * 60)
    print(f"평가 완료 — {name}  (stage={args.stage})")
    print("-" * 60)
    print(f"{'benchmark':35s} {'CER%':>7} {'sCER%':>7} {'n':>6}")
    print("-" * 60)
    for bid, r in results.items():
        print(f"{bid:35s} {r.cer:7.2f} {r.scer:7.2f} {r.samples:6d}")
    print("=" * 60)
    print(f"리포트: {out_dir}/")
    log.info("Evaluation finished", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
