#!/usr/bin/env bash
# 평가 런처 — conda 환경 + GPU 에서 scripts/eval.py 를 실행한다.
# (평가 yaml 엔 runtime 섹션이 없으므로 env/GPU 는 환경변수/기본값으로 잡는다)
#
# 사용:
#   scripts/eval.sh BENCHMARK/configs/eval/whisper_baseline.yaml          # GOLD 평가
#   scripts/eval.sh BENCHMARK/configs/eval/whisper_baseline.yaml --bench-root /data/ASR/BENCHMARK/SILVER
#   GPU=2 scripts/eval.sh <config> --benchmarks Sample10_PracticeRef --bench-root BENCHMARK/data
#
# 첫 인자(평가 yaml) 뒤의 모든 옵션은 scripts/eval.py 로 그대로 전달된다.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "사용법: scripts/eval.sh <eval_config.yaml> [scripts/eval.py 옵션...]" >&2
  exit 1
fi
CONFIG="$1"; shift

GPU="${GPU:-0}"
ENV="${ENV:-train-asr}"

echo "[eval] config=$CONFIG  gpu=$GPU  env=$ENV  args=$*"
echo "----------------------------------------------------------------"

CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV" --no-capture-output \
  python scripts/eval.py --config "$CONFIG" "$@"
rc=$?

echo "----------------------------------------------------------------"
echo "EXIT=$rc"
exit $rc
