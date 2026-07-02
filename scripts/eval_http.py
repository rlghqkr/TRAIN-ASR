"""HTTP ASR 서버 평가 (CLI).

로컬 HF 체크포인트가 아니라 **떠 있는 HTTP 서버**(whisper.cpp, Qwen3-ASR vLLM
래퍼 등 — AI-SPECTRUM 의 whisper 호환 `/inference` 엔드포인트)를 대상으로
`scripts/eval.py` 와 동일한 리포트(`evaluation_report.json` / `predictions.jsonl`)
를 만든다. 그래야 `compare_results.py` / `compare_samples.py` 를 그대로 재사용해
비교 엑셀을 뽑을 수 있다.

평가 로직 SoT 는 그대로 project/evaluation/evaluate_on_benchmark_suite — 이
스크립트는 "서버 URL → predict_fn(HTTP 동시요청) 조립 → 호출" 만 담당한다.
(scripts/eval.py 가 HF 모델용 predict_fn 을 조립하는 것과 동일한 역할 분담.)

사용법:
    python scripts/eval_http.py \
        --name qwen3_asr --server-url http://127.0.0.1:8102 \
        --benchmarks kiosk_cafe_order TTS_cafe_order_10speaker

    python scripts/eval_http.py \
        --name whisper_medium_cafe_live --server-url http://127.0.0.1:8084 \
        --benchmarks kiosk_cafe_order TTS_cafe_order_10speaker \
        --concurrency 16
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import structlog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = structlog.get_logger()

# whisper 호환 서버 대상 기본 벤치마크 데이터 루트. GOLD 샘플셋(리포지토리 다른 평가와 동일 소스).
DEFAULT_BENCH_DATA = "/data/ASR/BENCHMARK/GOLD"
TRANSCRIPT_NAME = "transcript.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTTP ASR 서버 벤치마크 평가")
    p.add_argument("--name", required=True,
                   help="결과 폴더명 (BENCHMARK/results/<name>/)")
    p.add_argument("--server-url", required=True,
                   help="whisper 호환 서버 URL (예: http://127.0.0.1:8084)")
    p.add_argument("--benchmarks", nargs="+", required=True,
                   help="평가할 bench_id 목록")
    p.add_argument("--bench-root", default=DEFAULT_BENCH_DATA,
                   help=f"벤치마크 데이터 루트. <root>/<bench_id>/transcript.jsonl. 기본 {DEFAULT_BENCH_DATA}")
    p.add_argument("--language", default="ko", help="언어 코드 (기본 ko)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="서버로 동시에 보낼 요청 수 (기본 8)")
    p.add_argument("--timeout", type=float, default=60.0, help="요청 타임아웃(초)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="evaluate_on_benchmark_suite 진행바 단위 배치 크기")
    p.add_argument("--no-timestamp", action="store_true",
                   help="결과 폴더명에 타임스탬프 안 붙임")
    return p.parse_args()


def resolve_benchmark_paths(bench_ids: list[str], *, bench_root: str) -> dict[str, str]:
    """bench_id 목록 → {bench_id: transcript.jsonl 절대경로}. 없으면 Fail Fast."""
    root = Path(bench_root)
    paths: dict[str, str] = {}
    for bid in bench_ids:
        path = root / bid / TRANSCRIPT_NAME
        if not path.exists():
            raise FileNotFoundError(f"벤치마크 파일 없음: {path}")
        paths[bid] = str(path)
    return paths


def build_predict_fn(server_url: str, *, language: str, timeout: float, concurrency: int):
    """서버 URL → predict_fn(오디오 경로 리스트 → 예측 텍스트 리스트).

    whisper.cpp / Qwen3-ASR 래퍼 둘 다 같은 whisper 호환 인터페이스
    (`POST /inference`, multipart `file` + `language` → JSON `{"text": ...}`)
    를 쓰므로 동일한 클라이언트로 둘 다 평가 가능.
    """
    client = httpx.Client(timeout=timeout, limits=httpx.Limits(max_connections=concurrency))

    def _transcribe_one(path: str) -> str:
        with open(path, "rb") as f:
            files = {"file": (Path(path).name, f, "audio/wav")}
            data = {"language": language, "response_format": "json"}
            resp = client.post(f"{server_url}/inference", files=files, data=data)
        if resp.status_code != 200:
            raise RuntimeError(
                f"ASR 서버 오류 ({resp.status_code}) path={path}: {resp.text[:300]}"
            )
        return resp.json()["text"].strip()

    def predict_fn(audio_paths: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(_transcribe_one, audio_paths))

    return predict_fn


def main() -> None:
    args = parse_args()

    name = args.name
    if not args.no_timestamp:
        import datetime
        name = f"{name}__{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"

    log = logger.bind(recognizer=name, server_url=args.server_url)

    # 헬스체크 — 서버가 안 떠 있으면 여기서 Fail Fast.
    health = httpx.get(f"{args.server_url}/health", timeout=10.0)
    if health.status_code != 200:
        raise RuntimeError(f"서버 헬스체크 실패 ({health.status_code}): {args.server_url}/health")
    log.info("Server health OK", response=health.json())

    benchmark_paths = resolve_benchmark_paths(args.benchmarks, bench_root=args.bench_root)
    predict_fn = build_predict_fn(
        args.server_url, language=args.language,
        timeout=args.timeout, concurrency=args.concurrency,
    )

    from project.evaluation import evaluate_on_benchmark_suite

    out_dir = ROOT / "BENCHMARK" / "results" / name
    log.info("Evaluation started", out_dir=str(out_dir), n_benchmarks=len(benchmark_paths))

    results = evaluate_on_benchmark_suite(
        model_name=name,
        predict_fn=predict_fn,
        benchmark_paths=benchmark_paths,
        out_dir=out_dir,
        batch_size=args.batch_size,
        save_diff=True,
        sample_frac=1.0,
    )

    print("\n" + "=" * 60)
    print(f"평가 완료 — {name}  ({args.server_url})")
    print("-" * 60)
    print(f"{'benchmark':35s} {'CER%':>7} {'sCER%':>7} {'n':>6}")
    print("-" * 60)
    for bid, r in results.items():
        print(f"{bid:35s} {r.cer:7.2f} {r.scer:7.2f} {r.samples:6d}")
    print("=" * 60)
    print(f"리포트: {out_dir}/")
    log.info("Evaluation finished", out_dir=str(out_dir))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("eval_http failed", error=str(e))
        sys.exit(1)
