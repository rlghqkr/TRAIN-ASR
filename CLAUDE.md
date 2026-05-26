# CLAUDE.md

이 파일은 Claude (및 다른 AI 코딩 어시스턴트)가 이 저장소에서 작업할 때
반드시 따라야 할 운영 규칙입니다. 새 작업을 시작하기 전에 항상 이 문서를 먼저 읽으세요.

---

## 1. 프로젝트 소개

- **이름**: TRAIN-ASR (+ BENCHMARK 하위 프로젝트)
- **목적**: 한국어 음성인식 모델(Whisper, SenseVoice 등) 학습 + 도메인·화자별
  벤치마크 평가
- **연구 영역**: 한국어 음성인식 (ASR)
- **메트릭**: CER (Character Error Rate) 우선, WER 보조
- **데이터 단계**: RAW → SILVER → GOLD (자세한 정책은 [GUIDELINE/04-data.md](GUIDELINE/04-data.md))
- **평가 단위**: 도메인(카페주문 / 콜센터 …) + 화자군(노인 / 어린이 …) + 환경(잡음 / 사투리 …) 별 벤치마크
- **문서 인덱스**: [GUIDELINE/README.md](GUIDELINE/README.md)
- **인턴 학습 자료**: [GUIDELINE/benchmark-and-training-guide.md](GUIDELINE/benchmark-and-training-guide.md)

---

## 2. 빠른 시작

```bash
# 1) conda 환경 생성 + 활성화
conda create -n <env-name> python=3.11 -y
conda activate <env-name>

# 2) 의존성 설치 (PyTorch는 환경 확인 후 requirements.txt 주석 해제)
make setup

# 3) 환경 변수
cp .env.example .env
# .env 편집 (W&B API key 등)

# 4) 학습 실행
python scripts/train.py --config configs/default.yaml
```

자주 쓰는 명령어:

```bash
make setup        # 의존성 + pre-commit 설치 (최초 1회)
make style        # 포맷 + 린트 자동 수정
make quality      # 포맷/린트 검사만 (CI에서 사용)
make test         # 테스트 실행
make typecheck    # mypy
make clean        # 캐시 제거
```

---

## 3. 프로젝트 구조

기본 골격은 최소화되어 있습니다. 프로젝트 규모에 맞춰 자유롭게 확장하세요.

```
.
├── configs/         # 실험 설정 (YAML)
├── data/            # 데이터 (커밋 안 됨, 출처는 data/README.md)
├── notebooks/       # 탐색/분석 (출력은 nbstripout로 자동 정리)
├── reports/         # 그림/리포트
├── scripts/         # 진입점 (train.py 등)
├── project/
│   └── utils/       # 재현성(seed), config 로더, W&B 헬퍼
└── tests/           # 핵심 함수 테스트 (가벼움)
```

### 권장 확장 (필요 시 직접 추가)

프로젝트가 커지면 `project/` 안에 다음 같은 서브패키지를 추가하면 깔끔합니다:

- `data/` — 로딩/전처리 (Dataset, DataLoader)
- `models/` — 모델 정의
- `training/` — 학습 루프, Trainer
- `evaluation/` — 평가 메트릭
- `viz/` — 시각화 헬퍼

데이터도 작은 프로젝트는 `data/` 하나로, 커지면 `raw/interim/processed/external/` 분리.

**원칙: 미리 빈 폴더를 만들지 말고, 실제로 코드가 들어갈 시점에 만든다.**

---

## 4. 핵심 코딩 원칙

| 원칙 | 설명 |
|------|------|
| **명시적 > 암시적** | 코드의 동작이 명확하게 드러나야 함 |
| **Fail Fast** | 문제가 있으면 즉시 에러 발생, 조용한 실패 금지 |
| **No Silent Fallback** | 암시적/자동 fallback 금지 |
| **Explicit Configuration** | 모든 설정은 명시적으로, 매직 넘버 금지 |

### 금지 패턴

```python
# BAD: 조용한 fallback
def get_item(item_id: str) -> Item:
    item = items.get(item_id)
    if item is None:
        return default_item  # 어떤 아이템이 사용되는지 모름

# GOOD: 명시적 에러
def get_item(item_id: str) -> Item:
    item = items.get(item_id)
    if item is None:
        raise ItemNotFoundError(f"Item '{item_id}' not found. Available: {list(items.keys())}")
    return item
```

```python
# BAD: 예외 무시
try:
    result = external_api.call()
except Exception:
    pass

# GOOD: 명시적 에러 처리
try:
    result = external_api.call()
except ExternalAPIError as e:
    logger.error("External API call failed", error=str(e))
    raise ServiceUnavailableError(f"External API failed: {e}") from e
```

```python
# BAD: 환경변수 기본값 fallback
api_url = os.getenv("API_URL", "http://localhost:8000")

# GOOD: 필수 환경변수 명시적 검증
api_url = os.environ["API_URL"]  # 미설정 시 KeyError 발생
```

---

## 5. Python 스타일

### 타입 힌트 (필수)

```python
def process_data(
    data_id: str,
    items: list[dict[str, str]],
    *,
    threshold: float = 0.7,
    limit: int | None = None,
) -> ProcessResult:
    ...
```

### 함수 설계

```python
# GOOD: 단일 책임, 명확한 이름
def validate_api_key(api_key: str) -> bool:
    """API 키 유효성 검증."""
    if not api_key:
        raise InvalidAPIKeyError("API key is required")
    if not api_key.startswith("sk-"):
        raise InvalidAPIKeyError("API key must start with 'sk-'")
    return True

# BAD: 모호한 이름, 조용한 실패
def check(key):
    if key:
        return True
    return False
```

### 클래스 설계

- 데이터 클래스: `@dataclass(frozen=True)` 사용 (불변성 권장)
- 추상 클래스: `ABC` 사용, `@abstractmethod` 명시

### Import 순서

```python
# 1. 표준 라이브러리
import os
from datetime import datetime

# 2. 서드파티 라이브러리
import numpy as np
import torch
from torch.utils.data import DataLoader

# 3. 로컬 모듈
from .config import settings
from .exceptions import AppError
```

---

## 6. 에러 처리

### 커스텀 예외 계층

```python
class AppError(Exception):
    """애플리케이션 기본 예외."""
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class ConfigurationError(AppError): pass
class DataError(AppError): pass
class ModelError(AppError): pass
class TrainingError(AppError): pass
```

### 에러 처리 패턴

```python
# GOOD: 구체적인 예외, 충분한 컨텍스트
def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise ModelError(
            f"Checkpoint not found: {path}",
            details={"path": str(path)},
        )
    try:
        return torch.load(path, map_location="cpu")
    except (OSError, RuntimeError) as e:
        raise ModelError(
            f"Failed to load checkpoint: {path}",
            details={"path": str(path), "error": str(e)},
        ) from e

# BAD: 포괄적 예외, 조용한 실패
def load_checkpoint(path):
    try:
        return torch.load(path)
    except Exception:
        return None  # 절대 금지
```

---

## 7. 로깅 (structlog)

이 프로젝트는 `structlog`을 사용합니다. `print` 대신 항상 logger를 사용하세요.

```python
import structlog

logger = structlog.get_logger()

def train_epoch(epoch: int, model_name: str) -> Metrics:
    log = logger.bind(epoch=epoch, model=model_name)
    log.info("Epoch started")

    try:
        metrics = run_training_loop()
        log.info(
            "Epoch completed",
            loss=metrics.loss,
            acc=metrics.accuracy,
            duration_sec=metrics.duration,
        )
        return metrics
    except TrainingError as e:
        log.error(
            "Epoch failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise
```

### 로깅 레벨

| 레벨 | 용도 |
|------|------|
| `DEBUG` | 개발 중 상세 정보 |
| `INFO` | 일반적인 작업 흐름 |
| `WARNING` | 잠재적 문제 |
| `ERROR` | 에러 발생 (복구 가능) |
| `CRITICAL` | 심각한 에러 (복구 불가) |

### 주의사항

- **민감 정보(API 키, 비밀번호, PII) 로깅 금지**
- 구조화된 로깅 사용 (문자열 포매팅 대신 key-value)
- 에러 발생 시 충분한 컨텍스트 포함

---

## 8. 문서화 (Google Style)

### Docstring

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    *,
    grad_clip: float | None = None,
) -> EpochMetrics:
    """1 epoch 학습을 수행.

    Args:
        model: 학습 대상 모델.
        loader: 학습 데이터 로더.
        optimizer: 옵티마이저.
        grad_clip: 그래디언트 클리핑 노름. None이면 클리핑 안 함.

    Returns:
        EpochMetrics: 평균 loss, accuracy, 소요 시간 등을 담은 객체.

    Raises:
        TrainingError: NaN loss 발생 또는 그래디언트 폭주 시.
    """
```

### 인라인 주석

```python
# GOOD: 왜(Why)를 설명
# torch.compile은 첫 호출 시 컴파일 비용이 크므로 warmup epoch 외부에서 호출
model = torch.compile(model)

# BAD: 무엇(What)을 설명 (코드로 이미 명확함)
# 모델을 컴파일
model = torch.compile(model)
```

### TODO/FIXME 규칙

```python
# TODO(username): 구체적인 작업 내용 - #이슈번호
# FIXME(username): 버그 설명 - #이슈번호
# NOTE: 중요한 참고 사항
# HACK: 임시 해결책 (반드시 이유 명시)
```

---

## 9. Git 커밋 컨벤션

### 형식

```
<type>: <subject>

<body>

<footer>
```

### Type

| Type | 설명 |
|------|------|
| `feat` | 새로운 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `style` | 코드 스타일 (포매팅 등) |
| `refactor` | 리팩토링 |
| `test` | 테스트 추가/수정 |
| `exp` | 실험 추가/수정 |
| `chore` | 빌드, 설정 변경 |

### 예시

```
feat: add cosine LR scheduler

- Add CosineAnnealingLR to training/scheduler.py
- Wire it through configs/default.yaml
- Update docs

Closes #12
```

```
exp: add wav2vec2 baseline experiment

configs/wav2vec2_base.yaml 추가. WER 18.2 → 14.7로 개선됨.
W&B run: https://wandb.ai/.../runs/abc123

Refs #34
```

---

## 10. 코드 수정 규칙

| 원칙 | 설명 |
|------|------|
| **요청 범위만 수정** | 사용자가 요청한 범위 내에서만 코드 변경 |
| **사전 컨펌 필수** | 추가 수정이 필요하면 수정 전에 컨펌 받기 |
| **기존 코드 존중** | 기존 주석, 스타일, 구조를 임의로 변경하지 않음 |

### 금지 사항

- 요청하지 않은 코드 정리/리팩토링
- 기존 주석 임의 삭제
- "더 깔끔해 보여서" 등의 이유로 임의 수정

### 허용 사항 (컨펌 후)

더 좋은 아이디어가 있을 경우:
1. 먼저 제안 내용 설명
2. 사용자 컨펌 받기
3. 컨펌 후 수정 진행

```
# BAD: 임의로 수정
사용자: "함수 A에 파라미터 추가해줘"
→ 함수 A 수정 + 주석 정리 + 다른 함수도 리팩토링

# GOOD: 요청 범위만 수정
사용자: "함수 A에 파라미터 추가해줘"
→ 함수 A에 파라미터만 추가

# GOOD: 추가 제안 시 컨펌
사용자: "함수 A에 파라미터 추가해줘"
→ "파라미터 추가하겠습니다. 추가로 관련 함수 B도 수정하면 좋을 것 같은데, 같이 수정할까요?"
→ 사용자 컨펌 후 진행
```

---

## 11. 연구 워크플로

### 새 실험 추가

1. **설정 파일 작성**: `configs/<exp_name>.yaml` 을 `configs/default.yaml` 기반으로 생성
2. **변경점 명시**: 설정 파일 상단 주석에 "이 실험에서 무엇을, 왜 바꾸는지" 적기
3. **실행**: `python scripts/train.py --config configs/<exp_name>.yaml`
4. **W&B 확인**: run name과 link을 PR/이슈에 기록
5. **commit**: `exp: ...` 형식으로 설정 파일과 함께 커밋

### 노트북 규칙

- 파일명: `<번호>_<설명>.ipynb` (예: `01_eda.ipynb`)
- **출력은 자동 제거됨** (`nbstripout` pre-commit hook)
- 같은 코드 두 번 이상 작성 시 → `project/` 모듈로 옮긴 뒤 import
- "노트북에서만 동작" 상태 금지: 의존성은 즉시 `requirements.txt`에 반영

### 데이터 다루기

- 원본 데이터는 read-only, 절대 수정 금지
- 새 데이터 추가 시 `data/README.md`에 출처/라이선스/받는 법 기록
- 큰 파일(수 GB+)은 git이 아닌 별도 스토리지(S3 등)에 두고 경로만 기록

### W&B 셋업

```bash
wandb login   # 최초 1회
# 또는 .env 에 WANDB_API_KEY 설정
```

`configs/<exp>.yaml` 의 `wandb` 섹션:

```yaml
wandb:
  enabled: true
  project: my-project
  entity: <팀/개인>
  tags: [baseline, v1]
  notes: "What's special about this run"
```

### 재현성

- 모든 학습 스크립트는 `seed_everything(seed)` 호출
- 설정 파일에 `experiment.seed` 명시
- 환경 변수, 패키지 버전은 lockfile(또는 `pip freeze`)로 박제

---

## 12. 보안 가이드라인

<!--
TODO: 회사/팀 보안 가이드라인이 정해지면 아래 내용을 채우거나 외부 문서로 링크하세요.

예시 항목:
- 사내 보안 가이드라인 문서 링크
- 보안 영역(인증/암호화/시크릿/PII/의존성 추가) 변경 프로세스
- 취약점 리포트 채널
- AI 어시스턴트가 자동 수정하지 말아야 할 영역 (Out of Scope)
-->

### 기본 원칙 (정책 확정 전 임시 가이드)

다음 영역은 **AI 어시스턴트가 자동 수정하지 않습니다**. 변경이 필요하면 코드를
수정하지 말고 사용자에게 보고만 하세요.

- 인증/인가 로직
- 시크릿 관리 (`.env`, API key, 토큰)
- 외부 통신/네트워크 설정
- 개인정보(PII) 처리
- `requirements*.txt` 의존성 추가/업데이트 (보안 패치 포함)
- `.github/workflows/` 시크릿/권한 설정

### 절대 금지

- 코드/문서/커밋 메시지에 실제 시크릿 값 포함
- `.env` 파일을 git에 커밋
- 민감 정보(API 키, 비밀번호, PII) 로깅
