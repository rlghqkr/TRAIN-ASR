# 2-A. 모델 학습 — Whisper

> 공통 부분(사전조건·공통 yaml·CLI·평가 연결)은 [2_모델학습.md](2_모델학습.md) 참조.
> 이 문서는 **Whisper 전용** 내용만 다룬다.

Whisper 는 HuggingFace `Seq2SeqTrainer` 기반이라 **노트북 셀이나 CLI 에서 바로 학습**된다.
학습 중 매 평가 스텝마다 **CER/sCER/WER** 를 계산해 best 모델을 고른다.

코드 진실원:
- 어댑터: [`../project/data/adapters/whisper.py`](../project/data/adapters/whisper.py) — `to_whisper_dataset`
- Trainer: [`../project/training/whisper.py`](../project/training/whisper.py) — `build_trainer`
- 오케스트레이션: [`../project/training/run.py`](../project/training/run.py) — `run_whisper_training`

---

## TODO 1. 학습 yaml — Whisper 키

공통 키 위에 `models.whisper` 만 추가하면 된다.

```yaml
models:
  whisper:
    backbone: openai/whisper-small   # HF ID 또는 로컬 경로. tiny/base/small/medium/large-v3
    language: ko                     # 강제 디코더 언어 토큰 (<|ko|>)
    task: transcribe                 # transcribe (그대로) | translate (영어로)
```

Whisper 가 참고하는 `training` 키 (공통 yaml 의 값들):

| 키 | 의미 |
|---|---|
| `epochs`, `batch_size`, `lr` | 표준 하이퍼파라미터 |
| `grad_accum_steps` | effective batch = `batch_size × grad_accum_steps × num_gpus` |
| `scheduler` | `cosine` / `linear` / `constant` |
| `warmup_steps`, `weight_decay`, `grad_clip` | 안정화 |
| `eval_every` / `save_every` / `log_every` | 스텝 주기 |
| `early_stopping_patience` | 설정 시 **CER 기준 best 모델** 추적 + 조기 종료 (`load_best_model_at_end`) |
| `precision` | `bf16` / `fp16` / `fp32` |

> `early_stopping_patience` 를 켜면 `save_every` 가 `eval_every` 의 배수로 자동 정렬된다(HF 제약).

---

## TODO 2. 어댑터 — HuggingFace Dataset 으로 변환

`Sample` JSONL → log-mel `input_features` + 토큰 `labels`.

```python
from project.data import load_samples
from project.data.adapters.whisper import to_whisper_dataset

train = load_samples("data/SILVER/train.jsonl")
val   = load_samples("data/SILVER/val.jsonl")

train_ds = to_whisper_dataset(train, backbone="openai/whisper-small")
val_ds   = to_whisper_dataset(val,   backbone="openai/whisper-small")
```

- `text_field` (기본 `text_norm`) 가 `labels` 의 원천. `text` 로 학습하려면 인자로 전달.
- 오디오 SR 이 16kHz 가 아니면 *에러* (Fail Fast) — 전처리에서 16kHz 통일 전제.

검증: `print(train_ds)` → `input_features`, `labels`, `id` 컬럼이 보이면 정상.

---

## TODO 3. 학습 실행

### 방법 A. CLI (권장 — 한 줄)

```bash
scripts/train.sh configs/<exp>.yaml whisper
```

런처가 config 의 `runtime.conda_env`/`cuda_visible_devices` 로 환경·GPU 를 잡고 학습을 실행한다.
내부적으로 `run_whisper_training()` 이 ①로드+검증 → ②어댑터 → ③Trainer → ④학습 → ⑤저장을 한 번에 수행한다.

> 런처 없이 직접: `CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/<exp>.yaml --model whisper`

### 방법 B. 노트북 셀에서 직접

```python
from project.utils import load_config
from project.training.whisper import build_trainer

cfg = load_config("configs/<exp>.yaml")
trainer = build_trainer(cfg, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
trainer.save_model()   # outputs/<exp>/ 에 HF 형식 저장
```

노트북 골격: [`../notebooks/11_train_whisper.ipynb`](../notebooks/11_train_whisper.ipynb)

검증: 로그에 `loss` 와 함께 `eval_cer` 가 내려가면 정상. (loss 만 보지 말 것)

### 재개

```yaml
training:
  resume_from: outputs/<exp>/checkpoint-<step>   # 중단된 체크포인트
```

---

## TODO 4. 산출물 & 평가 연결

`run_whisper_training` / `trainer.save_model()` 은 `outputs/<exp>/` 에 **HF 형식 디렉토리**로 저장(+ 재현용 `config.yaml`).

평가 yaml 의 `recognizer` (자세한 건 [공통 평가 TODO](2_모델학습.md) + [1번](1_벤치마크구축.md)):

```yaml
recognizer:
  name: <exp>_v1
  type: whisper
  model_path: outputs/<exp>      # ← HF 디렉토리 (프로세서 포함 → backbone 불필요)
  options:
    language: ko
    task: transcribe
    beam_size: 5

benchmarks:
  - Sample10_PracticeRef

batch_size: 16
sample_frac: 1.0

runtime:
  cuda_visible_devices: "0"
  conda_env: train-asr
```

> `model_path` 가 **디렉토리**면 모델·프로세서를 모두 거기서 로드한다 (학습 결과는 프로세서 포함).
> **단일 `.pt`** 가중치 파일일 때만 `backbone:` 을 추가해 processor/기본 아키텍처 출처를 지정한다
> ([`build_predict_fn`](../project/data/adapters/whisper.py)).

---

## Whisper 특이 함정

- **`.pt` 평가 시 backbone 누락** → 가중치만 든 `.pt` 를 평가하면 `backbone` 으로 processor 출처를 줘야 함 (폴더 평가는 불필요).
- **`predict_with_generate` 느림** → eval 스텝이 무거우면 `eval_every` 를 늘려 학습 throughput 확보.
- **language/task 미지정** → 한국어인데 디코더가 영어로 새면 `models.whisper.language: ko` 확인.
- **큰 LR** → 사전학습 능력 손상(catastrophic forgetting). 1e-6 ~ 5e-5 범위 유지.
