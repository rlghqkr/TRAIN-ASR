# TRAIN-ASR

> 한국어 음성인식(ASR) 모델(Whisper, SenseVoice 등) 학습·평가 파이프라인.

이 저장소는 **학습용 (TRAIN-ASR)** + **평가용 (BENCHMARK)** 두 부분으로 구성됩니다.

| 구분 | 위치 | 역할 |
|---|---|---|
| **TRAIN-ASR** | 저장소 루트 | 데이터 전처리 + 학습 + 모델 저장/로그 |
| **BENCHMARK** | [./BENCHMARK/](./BENCHMARK/) | 평가 데이터셋(벤치마크) 구축 + 모델 평가 (CER 기준) |

한국어는 **CER (Character Error Rate)** 을 표준 메트릭으로 사용합니다.
도메인·화자별 벤치마크(카페주문 / 노인 / 어린이 / 사투리 …) 로 잘게 나누어
평가합니다.

---

## 어디부터 읽을까?

| 누구 | 시작점 |
|---|---|
| **이 저장소 처음** | [GUIDELINE/README.md](GUIDELINE/README.md) — 작업 매뉴얼 인덱스 |
| **인턴 / 신규 합류자** | [GUIDELINE/benchmark-and-training-guide.md](GUIDELINE/benchmark-and-training-guide.md) — Phase 1 (벤치마크 구축) / Phase 2 (모델 학습) 단계별 가이드 |
| **학습 실험 하고 싶음** | [GUIDELINE/02-training.md](GUIDELINE/02-training.md) |
| **평가 / 벤치마크** | [GUIDELINE/03-benchmark.md](GUIDELINE/03-benchmark.md) |
| **데이터 정책 (RAW/SILVER/GOLD)** | [GUIDELINE/04-data.md](GUIDELINE/04-data.md) |
| **커밋·코딩 컨벤션** | [GUIDELINE/05-conventions.md](GUIDELINE/05-conventions.md) |
| **AI 어시스턴트 운영 규칙** | [CLAUDE.md](CLAUDE.md) |

---

## Quick Start

### 작업 환경 개요

| 항목 | 값 |
|---|---|
| GPU 서버 (JupyterLab) | `https://mpwav-gpu:50443/` |
| Python | 3.11 |

### 1. 서버 접속

브라우저로 `https://mpwav-gpu:50443/` 접속 → JupyterLab UI.

터미널이 필요하면 JupyterLab 의 **File → New → Terminal** 또는 SSH 로 같은 서버 접속.

### 2. conda 환경 생성 + 활성화 (최초 1회)

JupyterLab 터미널에서:

```bash
# 환경 생성 (이름은 자유롭게)
conda create -n <env-name> python=3.11 -y
conda activate <env-name>

# 빠른 검증
python --version       # Python 3.11.x
which python           # .../envs/<env-name>/bin/python
```

`configs/<exp>.yaml` 의 `runtime.conda_env` 를 같은 이름으로 갱신합니다.

### 3. PyTorch 설치 (CUDA 환경에 맞춰)

`requirements.txt` 의 torch 라인은 기본적으로 **주석 처리**되어 있습니다.
머신마다 CUDA 버전이 다르므로 본인 환경에 맞춰 설치:

```bash
# CUDA 12.8 (RTX 40 / H100 등)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (V100 / A100 일부)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 전용 (개발/디버깅용)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

자세한 옵션은 [PyTorch 공식 가이드](https://pytorch.org/get-started/locally/) 참조.

### 4. 나머지 의존성 설치

```bash
make setup
```

이 명령은:

- `requirements.txt` 설치 (jiwer, funasr, transformers, datasets, soundfile 등 포함)
- `requirements-dev.txt` 설치 (ruff, mypy, pytest)
- `pre-commit` 훅 설치
- git commit 메시지 템플릿 설정

설치 확인:

```bash
python -c "
import torch, transformers, funasr, jiwer, soundfile
print('torch:        ', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers: ', transformers.__version__)
print('funasr:       ', funasr.__version__)
print('jiwer:        ', jiwer.__version__)
"
```

### 5. 환경 변수

```bash
cp .env.example .env
# .env 편집 — WANDB_API_KEY 등 (사수 확인)
```

### 6. 학습 실행

브라우저로 JupyterLab 접속 후 노트북 실행:

- SenseVoice 학습: [notebooks/10_train_sensevoice.ipynb](notebooks/10_train_sensevoice.ipynb)
- Whisper 학습: [notebooks/11_train_whisper.ipynb](notebooks/11_train_whisper.ipynb)

노트북 첫 셀이 conda env / GPU 할당을 확인합니다.

또는 터미널에서 스크립트:

```bash
python scripts/train.py --config configs/default.yaml
```

모델별 학습 가이드:

- 공통 학습 절차: [GUIDELINE/02-training.md](GUIDELINE/02-training.md)
- 모델별 실행 예시: [GUIDELINE/benchmark-and-training-guide.md](GUIDELINE/benchmark-and-training-guide.md) §6 (Phase 2)

### 7. 평가 실행

```bash
# (예시 — 인터페이스 구현 후 갱신)
bash BENCHMARK/scripts/evaluate_all.sh \
    outputs/<exp>/model.pt.best \
    BENCHMARK/results/<exp>/
```

자세한 사용법: [GUIDELINE/03-benchmark.md](GUIDELINE/03-benchmark.md)

---

## 환경 관리 팁

| 상황 | 명령 |
|---|---|
| 환경 활성화 | `conda activate <env-name>` |
| 비활성화 | `conda deactivate` |
| 패키지 목록 보기 | `conda list` 또는 `pip list` |
| 환경 삭제 (재설치 시) | `conda env remove -n <env-name>` |
| 환경 스냅샷 저장 | `pip freeze > env-snapshot.txt` |

GPU 충돌이 우려되면 `configs/<exp>.yaml` 의 `runtime.cuda_visible_devices` 로
사용 GPU 를 명시합니다 (예: `"0,1"`).

---

## 프로젝트 구조 (한눈에)

```
TRAIN-ASR/
├── CLAUDE.md             # AI 어시스턴트 운영 규칙
├── README.md             # ← 이 파일
├── Makefile              # setup/style/quality/test/typecheck
├── pyproject.toml        # ruff / mypy / pytest 설정
├── requirements*.txt
│
├── configs/              # 실험 설정 YAML
├── data/                 # 데이터 (커밋 안 됨) — RAW / SILVER / GOLD
│
├── GUIDELINE/            # 작업 매뉴얼 (절차 / 규칙 / 인터페이스)
│   ├── 01-setup.md
│   ├── 02-training.md
│   ├── 03-benchmark.md
│   ├── 04-data.md
│   ├── 05-conventions.md
│   └── benchmark-and-training-guide.md
│
├── BENCHMARK/            # 평가 하위 프로젝트 (벤치마크 + 평가 코드)
│
├── notebooks/            # 탐색/분석 (출력 자동 정리)
├── project/              # 파이썬 코드 패키지 (flat layout)
│   └── utils/            #   - seed, config, W&B 헬퍼
├── reports/              # 그림/리포트 산출물
├── scripts/              # 진입점 (train.py 등)
└── tests/                # 가벼운 단위 테스트
```

---

## Development

```bash
make style       # 포맷 + 린트 자동 수정
make quality     # 포맷/린트 검사 (CI에서 사용)
make test        # 테스트 실행
make typecheck   # mypy
make clean       # 캐시 제거
```

---

## 기여 (Contributing)

1. **새 실험**: [GUIDELINE/02-training.md](GUIDELINE/02-training.md)
2. **새 벤치마크**: [GUIDELINE/03-benchmark.md](GUIDELINE/03-benchmark.md)
3. **문제 발생**: [GUIDELINE/02-training.md](GUIDELINE/02-training.md) (트러블슈팅 섹션)
4. **코딩 규칙**: [GUIDELINE/05-conventions.md](GUIDELINE/05-conventions.md) · [CLAUDE.md](CLAUDE.md)

PR 작성 시:

- 한 PR = 한 변경점 (가설 + 결과 비교 형태로)
- W&B run link, 평가 리포트 링크 본문에 명시
- 커밋 메시지는 Conventional Commits (`feat:`, `fix:`, `exp:` …)

---

## 라이선스 / 데이터

- 코드 라이선스는 별도 `LICENSE` 파일 참고 (있는 경우)
- 학습/평가 데이터는 출처별 라이선스 준수. 자세한 출처 기록: [data/README.md](data/README.md)
- 외부 공유 시 PII / 라이선스 검토 필수
