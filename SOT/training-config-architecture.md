# 학습/평가 Config 아키텍처 (SOT)

> 최종 확정: 2026-06-16. 코드 전수 추적으로 검증됨.
> 2026-06-22 개정: 스키마 리뉴얼(`text_norm` 등) + 런처(`scripts/train.sh`/`eval.sh`) + 평가 CLI(`scripts/eval.py`) 반영.
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
  text_field:      # text | text_norm  (학습 타겟; 평가와 일관되게 text_norm 권장)
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

> `runtime.conda_env` / `runtime.cuda_visible_devices` 는 **런처 `scripts/train.sh`·`scripts/eval.sh`
> 가 읽어** 해당 conda env + GPU 로 실행한다. (python 진입점 `scripts/train.py` 자체는 자기
> conda env 를 못 바꾸므로 런처가 담당. 직접 python 실행 시엔 셸에서 activate/`CUDA_VISIBLE_DEVICES` 지정.)

---

## 3. `BENCHMARK/configs/eval/*.yaml` — 평가 전용 (유지)

```yaml
recognizer:
  name:        # results/<name>/ 폴더명
  type:        # whisper | sensevoice — 어댑터 분기
  model_path:  # 백본 또는 학습 체크포인트 폴더 (= outputs/<exp>/)
  backbone:    # (whisper) processor 로드용
  options:     # build_predict_fn 키워드 인자 (language, task, beam_size, device …)
benchmarks:    # [<bench_id>, ...] → <bench_root>/<bench_id>/transcript.jsonl 자동 lookup
batch_size:
```

- `benchmarks` 는 **경로가 아니라 ID 리스트** (ID 기반 자동 lookup).
- 읽는 곳: `scripts/eval.py`(CLI 진입점, `scripts/eval.sh` 가 호출), `notebooks/12_eval.ipynb`.
- ID → `transcript.jsonl` 경로 해석: `--stage gold|silver`(기본 gold) 또는 `--bench-root` 로.

---

## 4. 데이터 경로 규약

- **학습셋**: `paths.train_jsonl` / `val_jsonl` 에 직접 지정. GOLD 든 SILVER 든
  사용자가 가리키는 경로를 그대로 읽는다 (GOLD 전용 강제 없음).
- **평가셋**: `<bench_root>/<bench_id>/transcript.jsonl` (Sample 스키마).
  - 기본 `bench_root` = `/data/ASR/BENCHMARK/<STAGE>` (STAGE = `gold`(기본) | `silver`).
    GOLD 는 SILVER 에서 샘플링한 부분집합.
  - `BENCHMARK/data/<bench_id>` 는 `/data/ASR/BENCHMARK/SILVER/<bench_id>/` 로의
    **심볼릭 링크**(+ 커밋된 로컬 샘플 `Sample10_PracticeRef`). `BENCHMARK/data/` 는 `.gitignore`.
- 사용자 홈 절대경로(`/home/<user>/...`)를 **코드에 박지 않는다**. 경로는 config 로 뺀다.

---

## 5. 실행 환경

- conda env: **`train-asr`** (Whisper·SenseVoice 공용 단일 환경).
  `configs/default.yaml` 의 `runtime.conda_env` 메모도 이 값.
- Whisper 학습 deps: `transformers, accelerate, datasets, soundfile, jiwer` (+ torch).
  학습 중 eval 스텝마다 CER 을 계산하므로 **jiwer 필요** (`project.evaluation.compute_cer`).
- SenseVoice 학습 deps: `funasr` (torchrun 외부 실행).
- 로깅: `wandb` (`wandb.enabled: true` 일 때만 필요. `false` 면 no-op).

---

## 6. 학습 / 평가 실행 (요약)

런처가 config 의 `runtime`(conda_env/GPU)을 읽어 실행한다.

```bash
# 학습 — Whisper (HF Trainer 바로 학습)
scripts/train.sh configs/default.yaml whisper

# 학습 — SenseVoice (torchrun 명령만 출력 → 별도 tmux 에서 실행)
scripts/train.sh configs/default.yaml sensevoice

# 학습 후 자동 평가 (opt-in): outputs/<exp> 를 그 평가 yaml 로 평가
EVAL_CONFIG=BENCHMARK/configs/eval/whisper_baseline.yaml \
    scripts/train.sh configs/default.yaml whisper

# 평가만 — 기본 GOLD (가끔 --stage silver)
scripts/eval.sh BENCHMARK/configs/eval/whisper_baseline.yaml
```

- 학습 로직 SoT = `project/training/run.py`, 평가 진입점 = `scripts/eval.py`
  (`project.evaluation.evaluate_on_benchmark_suite`). CLI·노트북 모두 이들을 호출만 한다.
- 노트북: 학습 `notebooks/11_train_whisper.ipynb`, 평가 `notebooks/12_eval.ipynb`.
- 선결 조건: `paths.train_jsonl` / `val_jsonl` 가 실제로 존재해야 함.
- 직접 python 실행: `CUDA_VISIBLE_DEVICES=N python scripts/train.py --config ... --model whisper`.
