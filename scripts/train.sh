#!/usr/bin/env bash
# 학습 런처 — config 의 runtime 섹션(conda_env, cuda_visible_devices)을 읽어
# 해당 conda 환경 + GPU 에서 scripts/train.py 를 실행한다.
#
# python 진입점(scripts/train.py)은 자기 자신의 conda 환경을 바꿀 수 없으므로,
# "어느 환경/어느 GPU 에서 돌릴지"는 이 셸 런처가 책임진다.
#
# 사용:
#   scripts/train.sh                                  # configs/default.yaml, whisper
#   scripts/train.sh configs/smoke.yaml whisper       # config + model 지정
#   scripts/train.sh configs/default.yaml sensevoice  # SenseVoice (torchrun 명령 출력)
#   GPU=0 ENV=train-asr scripts/train.sh configs/smoke.yaml whisper   # 값 덮어쓰기
#
# 학습 후 자동 평가 (opt-in): EVAL_CONFIG 를 주면 학습 성공 시 그 평가 yaml 로
# outputs/<exp> 를 벤치마크에 평가한다. STAGE 로 gold/silver 선택(기본 gold).
#   EVAL_CONFIG=BENCHMARK/configs/eval/whisper_baseline.yaml \
#       scripts/train.sh configs/default.yaml whisper
#
# 우선순위: 환경변수(GPU/ENV) > config 의 runtime 값 > 기본값
set -uo pipefail

# 저장소 루트로 이동 (어디서 실행해도 동작)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-configs/default.yaml}"
MODEL="${2:-whisper}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config 파일 없음: $CONFIG" >&2
  exit 1
fi

# config 의 runtime 섹션에서 conda_env / cuda_visible_devices 추출.
# (이 시점엔 아직 학습 env 가 아니므로, yaml 파싱은 현재 셸의 python 사용)
read -r CFG_GPU CFG_ENV < <(python - "$CONFIG" <<'PY'
import sys, yaml
r = (yaml.safe_load(open(sys.argv[1])) or {}).get("runtime", {})
print(r.get("cuda_visible_devices", "0"), r.get("conda_env", "train-asr"))
PY
)

# 환경변수로 덮어쓸 수 있게 (없으면 config 값)
GPU="${GPU:-$CFG_GPU}"
ENV="${ENV:-$CFG_ENV}"

echo "[train] config=$CONFIG  model=$MODEL  gpu=$GPU  env=$ENV"
echo "----------------------------------------------------------------"

CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV" --no-capture-output \
  python scripts/train.py --config "$CONFIG" --model "$MODEL"
rc=$?

# 학습 후 자동 평가 (EVAL_CONFIG 지정 시에만, 학습 성공 시에만)
if [[ $rc -eq 0 && -n "${EVAL_CONFIG:-}" ]]; then
  EXP=$(python - "$CONFIG" <<'PY'
import sys, yaml
print((yaml.safe_load(open(sys.argv[1])) or {}).get("experiment", {}).get("name", ""))
PY
)
  if [[ -z "$EXP" ]]; then
    echo "[train→eval] experiment.name 을 못 읽어 평가 건너뜀" >&2
  else
    echo ""
    echo "[train→eval] EVAL_CONFIG=$EVAL_CONFIG  exp=$EXP  stage=${STAGE:-gold}"
    GPU="$GPU" ENV="$ENV" "$REPO_ROOT/scripts/eval.sh" "$EVAL_CONFIG" \
      --model-path "outputs/$EXP" --name "$EXP" --stage "${STAGE:-gold}"
    rc=$?
  fi
fi

echo "----------------------------------------------------------------"
echo "EXIT=$rc"
exit $rc
