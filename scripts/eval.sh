#!/usr/bin/env bash
# 평가 런처 — config 의 runtime 섹션(conda_env, cuda_visible_devices)을 읽어
# 해당 conda 환경 + GPU 에서 scripts/eval.py 를 실행한다. (scripts/train.sh 와 동일 방식)
#
# GPU 선택은 이 런처가 CUDA_VISIBLE_DEVICES 로 마스킹하고, 코드는 논리 'cuda:0' 만 쓴다.
#
# 사용:
#   scripts/eval.sh BENCHMARK/configs/whisper_baseline.yaml          # GOLD 평가
#   scripts/eval.sh BENCHMARK/configs/whisper_baseline.yaml --bench-root /data/ASR/BENCHMARK/SILVER
#   GPU=2 scripts/eval.sh <config> --benchmarks Sample10_PracticeRef --bench-root BENCHMARK/data
#   scripts/eval.sh BENCHMARK/configs/whisper_baseline.yaml --sample-frac 0.1   # 벤치마크별 10% 만 빠르게
#
# 첫 인자(평가 yaml) 뒤의 모든 옵션은 scripts/eval.py 로 그대로 전달된다.
#
# 우선순위: 환경변수(GPU/ENV) > config 의 runtime 값 > 기본값
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "사용법: scripts/eval.sh <eval_config.yaml> [scripts/eval.py 옵션...]" >&2
  exit 1
fi
CONFIG="$1"; shift

# config 의 runtime 섹션에서 cuda_visible_devices / conda_env 추출 (train.sh 와 동일).
read -r CFG_GPU CFG_ENV < <(python - "$CONFIG" <<'PY'
import sys, yaml
r = (yaml.safe_load(open(sys.argv[1])) or {}).get("runtime", {})
print(r.get("cuda_visible_devices", "0"), r.get("conda_env", "train-asr"))
PY
)

# 환경변수로 덮어쓸 수 있게 (없으면 config 값)
GPU="${GPU:-$CFG_GPU}"
ENV="${ENV:-$CFG_ENV}"

echo "[eval] config=$CONFIG  gpu=$GPU  env=$ENV  args=$*"
echo "----------------------------------------------------------------"

CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV" --no-capture-output \
  python scripts/eval.py --config "$CONFIG" "$@"
rc=$?

echo "----------------------------------------------------------------"
echo "EXIT=$rc"
exit $rc
