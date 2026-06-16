# 학습/평가 Config 아키텍처 (SOT)

> 최종 확정: 2026-06-16. 코드 전수 추적으로 검증됨.
> 충돌 시 이 문서가 기준. (구버전 `docs/training/configs.md` 는 현행 코드와 불일치 — 참고 금지)

---

## 1. 핵심 결정 — 학습 config 와 평가 config 를 분리한다

| 단계 | config 위치 | 소유 범위 |
|------|-------------|-----------|
| **학습** | `configs/default.yaml` | 학습셋 경로 + 모델 + 하이퍼파라미터 |
| **평가** | `BENCHMARK/configs/eval/*.yaml` | 평가셋(benchmarks) + `recognizer.model_path` |

- 두 config 는 서로를 모른다. **유일한 연결 고리는 체크포인트 경로** —
  학습이 `outputs/<exp>/` 를 만들고, 평가 yaml 의 `recognizer.model_path` 가 그걸 가리킨다.
- `configs/default.yaml` 에는 **평가 관련 키를 두지 않는다** (과거엔 `eval.*` 섹션과
  `paths.test_dir` 등이 있었으나 BENCHMARK 와 중복이라 제거함).
- 전처리(RAW→SILVER→GOLD)는 **이 리포 밖 별도 노트북**에서 한다. 따라서
  `configs/default.yaml` 에 RAW/SILVER/메타시트 경로를 두지 않는다.

### 왜

- 학습할 때 평가/전처리 노브가 같이 보이는 노이즈 제거.
- 평가는 이미 `BENCHMARK/configs/eval/*.yaml` 이 self-contained 하게 소유 중
  (`notebooks/00_build_benchmark.ipynb` / `BENCHMARK/README.md §3` 이 읽음).
- 학습/평가가 코드 레벨에선 이미 분리돼 있었고, config 만 한 덩어리였던 것을 정리.

---

## 2. `configs/default.yaml` — 학습 전용. 키별 실제 사용처

학습 진입점이 **실제로 읽는** 키만 둔다. (`scripts/train.py` →
`project/training/run.py`·`whisper.py`·`sensevoice.py` 추적 기준)

```yaml
experiment:
  name:            # outputs/<name>/ 폴더명 + W&B run name
  seed:            # seed_everything()

paths:
  repo_root:       # 절대경로 단일 앵커
  train_jsonl:     # 학습셋. silver/gold 어디든 직접 지정 (repo_root 기준 보간)
  val_jsonl:       # 검증셋 (학습 중 모니터링)
  outputs_dir:     # 학습 산출물 루트

models:
  whisper:    {backbone, language, task}
  sensevoice: {backbone, trust_remote_code}

data:
  text_field:      # text | text_normalized
  sample_rate:     # 16000
  max_label_len:   # 토큰 최대 길이
  num_workers:     # DataLoader 워커
  # (선택) sensevoice_train_jsonl / sensevoice_val_jsonl
  #        — SenseVoice 어댑터 변환 JSONL. 없으면 train_jsonl/val_jsonl 로 fallback.

training:          # epochs, batch_size, grad_accum_steps, lr, warmup_steps,
                   # weight_decay, scheduler, precision, grad_clip,
                   # log_every, eval_every, save_every, early_stopping_patience,
                   # (선택) batch_size_tokens, resume_from

wandb:             # enabled, project, entity, group, tags, notes
runtime:           # cuda_visible_devices, conda_env
```

### 모델별로 읽는 키

- **공통**: `experiment.*`, `paths.{train_jsonl,val_jsonl,outputs_dir}`, `data.*`,
  `training.*`
- **Whisper**: `models.whisper.*`, `wandb.*` (HF Trainer 가 W&B 직접 처리)
- **SenseVoice**: `models.sensevoice.*`, `runtime.cuda_visible_devices`(torchrun GPU),
  `training.batch_size_tokens`(선택), `data.sensevoice_*_jsonl`(선택)

### 제거한 키 (아무도 안 읽음 — 되살리지 말 것)

- `paths`: `data_root, raw_dir, silver_dir, gold_dir, corpora_meta_xlsx,
  benchmark_root, benchmark_silver_dir, benchmark_gold_dir, test_dir,
  benchmark_meta_xlsx, eval_results_dir`
- `data.max_audio_sec` — 코드 사용처 0
- `eval.*` 섹션 전체 — 평가는 BENCHMARK 소유

> 주의: `runtime.conda_env` 는 코드가 읽지 않는 **사람용 메모**다 (어느 env 를
> activate 할지 안내). 값이 틀려도 학습엔 영향 없음.

---

## 3. `BENCHMARK/configs/eval/*.yaml` — 평가 전용 (유지)

```yaml
recognizer:
  name:        # results/<name>/ 폴더명
  type:        # whisper | sensevoice — 어댑터 분기
  model_path:  # 백본 또는 학습 체크포인트 폴더 (= outputs/<exp>/)
  backbone:    # (whisper) processor 로드용
  options:     # build_predict_fn 키워드 인자 (language, task, beam_size, device …)
benchmarks:    # [<bench_id>, ...] → BENCHMARK/<bench_id>/samples.jsonl 자동 lookup
batch_size:
```

- `benchmarks` 는 **경로가 아니라 ID 리스트** (ID 기반 자동 lookup).
- 읽는 곳: `notebooks/00_build_benchmark.ipynb`, `BENCHMARK/README.md §3`.

---

## 4. 데이터 경로 규약

- **학습셋**: `paths.train_jsonl` / `val_jsonl` 에 직접 지정. GOLD 든 SILVER 든
  사용자가 가리키는 경로를 그대로 읽는다 (GOLD 전용 강제 없음).
- **평가셋**: `BENCHMARK/<bench_id>/samples.jsonl` (Sample 스키마).
  `BENCHMARK/data/<name>` 은 외부 스토리지(`/data/ASR/BENCHMARK/SILVER/...`)로의
  **심볼릭 링크**이며 `.gitignore` 됨 (`BENCHMARK/data/`).
- 사용자 홈 절대경로(`/home/<user>/...`)를 **코드에 박지 않는다**. 경로는 config 로 뺀다.

---

## 5. 실행 환경

- conda env: **`train-asr`** (Whisper·SenseVoice 공용 단일 환경).
  `configs/default.yaml` 의 `runtime.conda_env` 메모도 이 값.
- Whisper 학습 deps: `transformers, accelerate, datasets, soundfile` (+ torch).
  CER 은 자체 normalize 라 jiwer 불필요.
- SenseVoice 학습 deps: `funasr` (torchrun 외부 실행).

---

## 6. 학습 실행 (요약)

```bash
# Whisper — HF Trainer 바로 학습
CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
    --config configs/default.yaml --model whisper

# SenseVoice — torchrun 명령만 출력 → 별도 tmux 에서 실행
python scripts/train.py --config configs/default.yaml --model sensevoice
```

- 로직 SoT 는 `project/training/run.py`. CLI 도 노트북도 이 함수를 호출만 한다.
- 노트북 `notebooks/11_train_whisper.ipynb` 는 **학습까지**만 담당 (평가 셀 없음).
- 선결 조건: `paths.train_jsonl` / `val_jsonl` 가 실제로 존재해야 함.
