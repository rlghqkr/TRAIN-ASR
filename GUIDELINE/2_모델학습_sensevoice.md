# 2-B. 모델 학습 — SenseVoice

> 공통 부분(사전조건·공통 yaml·CLI·평가 연결)은 [2_모델학습.md](2_모델학습.md) 참조.
> 이 문서는 **SenseVoice 전용** 내용만 다룬다.

> 🚧 **상태**: 학습 *명령 생성* 까지 구현됨. FunASR 외부 프로세스 학습은 end-to-end 미검증.
> 실제 학습/평가가 한 번 돌면 본 문서 업데이트.

SenseVoice 는 FunASR 의 `train_ds.py` 를 **`torchrun` 외부 프로세스**로 실행한다.
HF Trainer 처럼 셀에서 바로 돌지 않고, **명령 문자열을 출력 → 별도 터미널(tmux)에서 실행**하는 흐름.

코드 진실원:
- 어댑터: [`../project/data/adapters/sensevoice.py`](../project/data/adapters/sensevoice.py) — `to_sensevoice_jsonl`
- 명령 빌더: [`../project/training/sensevoice.py`](../project/training/sensevoice.py) — `build_torchrun_command`
- 참고: `practice/2601_sensevoice_train/finetune.sh`

---

## TODO 1. 학습 yaml — SenseVoice 키

공통 키 위에 `models.sensevoice` 추가.

```yaml
models:
  sensevoice:
    backbone: /home/cssong/workspace/data/FM/MODEL/SenseVoiceSmall   # 로컬 백본 경로
    trust_remote_code: true        # FunASR 가 백본 폴더 안 .py 코드를 신뢰·실행
```

SenseVoice 가 참고하는 키:

| 키 | 의미 |
|---|---|
| `runtime.cuda_visible_devices` | GPU 인덱스. 개수로 `--nproc_per_node` 자동 결정 |
| `training.batch_size_tokens` | **토큰 단위** 동적 배칭 크기. 없으면 `batch_size × 1000` |
| `training.lr`, `training.epochs` | `optim_conf.lr`, `train_conf.max_epoch` 로 전달 |
| `data.sensevoice_train_jsonl` / `_val_jsonl` | 어댑터로 변환한 FunASR JSONL 경로 (없으면 `paths.train_jsonl` 사용) |

> ⚠️ Whisper 의 `batch_size` 와 의미가 다르다. SenseVoice 는 **토큰 수 기반** 동적 배칭.

---

## TODO 2. 어댑터 — FunASR JSONL 로 변환

`Sample` JSONL → FunASR 가 요구하는 한 줄 스키마(`key`/`source`/`source_len`/`target`/`target_len` + 고정 토큰들).

```python
from project.data import load_samples
from project.data.adapters.sensevoice import to_sensevoice_jsonl

train = load_samples("data/SILVER/train.jsonl")
val   = load_samples("data/SILVER/val.jsonl")

to_sensevoice_jsonl(train, "data/SILVER/adapted/sv/train.jsonl")
to_sensevoice_jsonl(val,   "data/SILVER/adapted/sv/val.jsonl")
```

- `text_field` (기본 `text_norm`) 가 `target` 의 원천.
- 변환된 경로를 yaml `data.sensevoice_train_jsonl` / `data.sensevoice_val_jsonl` 에 적어둔다.

검증: 출력 JSONL 첫 줄에 `source`(절대 wav 경로), `target`, `source_len`, `<|ko|>` 등이 보이면 정상.

---

## TODO 3. 학습 실행 — torchrun

### 방법 A. CLI (명령 출력)

```bash
scripts/train.sh configs/<exp>.yaml sensevoice
```

→ `torchrun ... train_ds.py ...` 멀티라인 명령이 **출력만** 된다. 이걸 복사해 tmux 에서 실행.

> 런처 없이 직접: `python scripts/train.py --config configs/<exp>.yaml --model sensevoice`

### 방법 B. 노트북에서 명령 생성

```python
from project.utils import load_config
from project.training.sensevoice import build_torchrun_command, print_command

cfg = load_config("configs/<exp>.yaml")
print_command(build_torchrun_command(cfg))   # 출력된 명령을 tmux 에서 실행
```

출력 예 (요지):

```bash
export CUDA_VISIBLE_DEVICES="0"
torchrun --nnodes 1 --nproc_per_node 1 --master_port 26669 <funasr>/bin/train_ds.py \
    ++model="...SenseVoiceSmall" \
    ++train_data_set_list="data/SILVER/adapted/sv/train.jsonl" \
    ++valid_data_set_list="data/SILVER/adapted/sv/val.jsonl" \
    ++dataset_conf.batch_size=8000 ++dataset_conf.batch_type="token" \
    ++train_conf.max_epoch=5 ++optim_conf.lr=1e-05 \
    ++output_dir="outputs/<exp>"
```

노트북 골격: [`../notebooks/10_train_sensevoice.ipynb`](../notebooks/10_train_sensevoice.ipynb)

> tmux 권장. 백그라운드 `run_in_background()` 도 있으나 노트북 커널이 꺼지면 같이 죽는다.

### 재개

FunASR 는 `output_dir` 의 마지막 체크포인트에서 자동 재개(같은 `output_dir` 로 재실행).

---

## TODO 4. 산출물 & 평가 연결

학습 결과는 `outputs/<exp>/` 에 **FunASR 형식**으로 저장된다.

평가 yaml 의 `recognizer` (자세한 건 [공통 평가 TODO](2_모델학습.md) + [1번](1_벤치마크구축.md)):

```yaml
recognizer:
  name: <exp>_v1
  type: sensevoice
  model_path: outputs/<exp>      # ← 학습된 SenseVoice 체크포인트
  options: {}                    # 옵션 없음 (FunASR 내부 처리). GPU 는 runtime 으로

benchmarks:
  - Sample10_PracticeRef

batch_size: 16
sample_frac: 1.0

runtime:
  cuda_visible_devices: "0"      # GPU 번호 (nvidia-smi 기준)
  conda_env: train-asr
```

> Whisper 와 달리 `backbone` 별도 지정 불필요 (FunASR `AutoModel` 이 폴더에서 로드). GPU 는 `runtime` 으로.

---

## SenseVoice 특이 함정

- **`batch_size` 를 Whisper 감각으로 설정** → 토큰 단위라 의미가 다름. `batch_size_tokens` 로 명시.
- **어댑터 변환 건너뜀** → 원본 `Sample` JSONL 을 그대로 넣으면 FunASR 가 못 읽음. 반드시 `to_sensevoice_jsonl`.
- **`funasr` 미설치** → 명령 빌드 시 `train_ds.py` 경로 탐색에서 에러. `pip install funasr`.
- **노트북 셀에서 직접 학습 시도** → SenseVoice 는 외부 프로세스. 셀은 *명령 생성* 까지만.
- **`No module named 'model'` 경고** → 백본 폴더 .py 의존성 경고. 추론/학습 동작엔 영향 없음.
