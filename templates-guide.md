# 사내 GitHub 템플릿 가이드

이 문서는 사내에서 운영하는 두 GitHub 템플릿(`template-research`, `template-dev`)의
**사용 방법 + 설계 결정**을 정리한 것입니다.

---

## 한 줄 요약

| 템플릿 | 언제 쓰나요 |
|---|---|
| **`template-research`** | ML/DL 실험·연구·데이터 분석. 학습/평가/노트북 작업 중심. |
| **`template-dev`** | 애플리케이션·서비스·CLI 도구·라이브러리. 코드 품질·테스트·배포 중심. |

새 레포를 만들 때 둘 중 하나를 고르세요. 회색지대(데이터 파이프라인, ML 서비스 등)는 **주된 활동**이 무엇인가로 판단:

- 실험·논문·분석이 주 → `research`
- 배포·운영·테스트가 주 → `dev`

---

## 사용 흐름

1. 사내 GitHub Organization에서 템플릿 레포로 이동
2. `Use this template` 버튼 클릭 → 새 레포 생성
3. 새 레포 clone
4. `README.md` 의 **First-time Setup Checklist** 항목 따라 초기 셋업
5. 체크리스트 섹션 삭제 후 첫 커밋

---

## 공통 설계 결정 (양쪽 템플릿)

### 패키지 관리: conda + pip 조합

`conda` 는 **Python 환경 격리에만** 사용하고, 패키지는 `pip + requirements.txt` 로
관리합니다.

- **conda 격리의 이점**: PyTorch/CUDA 같은 시스템 의존성 호환에 안전
- **pip 패키지의 이점**: 의존성 명시가 협업자에게 친숙, `requirements.txt` 는 어디서든 통용

```bash
conda create -n <env-name> python=3.11 -y
conda activate <env-name>
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### CLAUDE.md = AI 어시스턴트 운영 규칙

`CLAUDE.md` 파일이 양쪽 템플릿의 핵심 자산입니다. AI 코딩 어시스턴트(Claude, Cursor 등)가
이 저장소에서 작업할 때 따라야 할 규칙을 한 문서로 정리. 12~13개 섹션으로 구성:

1. 프로젝트 소개
2. 빠른 시작
3. 프로젝트 구조
4. 핵심 코딩 원칙 (Fail Fast, No Silent Fallback, Explicit Configuration)
5. Python 스타일 (타입 힌트 필수, Google docstring)
6. 에러 처리 (커스텀 예외 계층)
7. 로깅 (structlog)
8. **비동기 패턴** (`dev` 만)
9. 문서화
10. Git 커밋 컨벤션 (Conventional Commits)
11. 코드 수정 규칙 (요청 범위만, 사전 컨펌)
12. 워크플로 (research/dev 별 다름)
13. 보안 가이드라인 (TBD — 회사 정책 자리)

새 멤버는 새 레포 받자마자 이 파일 한 번 읽으면 사내 코딩 표준을 거의 다 흡수할 수 있습니다.

### 도구 셋업 (양쪽 동일)

- **포매터/린터**: `ruff` (line-length 119, 더블쿼트, isort 통합)
- **타입 체크**: `mypy`
- **테스트**: `pytest` (+ dev는 `pytest-asyncio`)
- **pre-commit**: ruff + 위생 hooks (`detect-private-key`, `check-yaml`, `check-toml`, `check-added-large-files` 등) + nbstripout (research)
- **CI**: GitHub Actions, `ubuntu-latest`, pip 캐시 적용

### Conventional Commits

`.gitmessage` 템플릿 자동 적용. 커밋 메시지 형식:
```
<type>: <subject>

<body>

<footer>
```

| Type | 설명 |
|------|------|
| `feat` | 새로운 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `style` | 코드 스타일 |
| `refactor` | 리팩토링 |
| `test` | 테스트 |
| `exp` | 실험 (research 전용) |
| `chore` | 빌드/설정 |

---

## 핵심 코딩 철학 (CLAUDE.md 요약)

| 원칙 | 설명 |
|------|------|
| **명시적 > 암시적** | 코드의 동작이 명확하게 드러나야 함 |
| **Fail Fast** | 문제가 있으면 즉시 에러 발생, 조용한 실패 금지 |
| **No Silent Fallback** | 암시적/자동 fallback 금지 |
| **Explicit Configuration** | 모든 설정은 명시적으로, 매직 넘버 금지 |

### 금지 패턴

```python
# BAD: 조용한 fallback
item = items.get(item_id)
if item is None:
    return default_item

# GOOD: 명시적 에러
item = items.get(item_id)
if item is None:
    raise NotFoundError(f"Item '{item_id}' not found")
```

```python
# BAD: 환경변수 fallback
api_url = os.getenv("API_URL", "http://localhost:8000")

# GOOD: 필수값 검증 (또는 dev 템플릿의 pydantic-settings 활용)
api_url = os.environ["API_URL"]   # 미설정 시 KeyError
```

---

## `template-research` 고유 결정

### Flat layout (`project/` 직속)

연구 코드는 PyPI 배포 대상이 아니므로 `src/` 한 단계를 두지 않고 `project/` 를
바로 root에 두는 flat layout 채택.

```
template-research/
├── configs/         # 실험 설정 (YAML)
├── data/            # 데이터 (커밋 안 됨, 출처는 data/README.md)
├── notebooks/       # 탐색/분석 (출력은 nbstripout로 자동 정리)
├── reports/         # 그림/리포트
├── scripts/         # 진입점 (train.py 등)
├── project/
│   └── utils/       # 재현성(seed), config 로더, W&B 헬퍼
└── tests/
```

### 권장 확장 구조 (필요 시 직접 추가)

기본은 `project/utils/` 만 두고, 프로젝트가 커지면 다음 같은 서브패키지를
**그때 추가**합니다.

- `project/data/` — 로딩/전처리 (Dataset, DataLoader)
- `project/models/` — 모델 정의
- `project/training/` — 학습 루프, Trainer
- `project/evaluation/` — 평가 메트릭
- `project/viz/` — 시각화 헬퍼

데이터 폴더도 작은 분석은 `data/` 하나, 커지면 `raw/interim/processed/external/` 분리.

> 원칙: 미리 빈 폴더를 만들지 말고, **실제로 코드가 들어갈 시점에 만든다.**

### 실험 추적: W&B

`project/utils/wandb_utils.py` 헬퍼로 init/finish 표준화. 사용 안 하려면
`configs/*.yaml` 의 `wandb.enabled: false` 로 끔.

### 설정 관리: YAML + argparse (단순)

Hydra 같은 무거운 도구 대신 표준 라이브러리만 사용. 작은 실험에 가볍게 시작.

### PyTorch는 주석 처리

서버/머신마다 CUDA 버전이 다름. `requirements.txt` 의 torch 라인은 기본적으로
주석 처리되어 있고, 본인 환경 확인 후 주석 해제 또는 PyTorch 공식 가이드 명령어로
별도 설치.

### 노트북 위생

- pre-commit `nbstripout` hook 으로 출력 자동 제거
- 같은 코드 두 번 이상 → `project/` 모듈로 옮긴 뒤 import

---

## `template-dev` 고유 결정

### src layout (`src/project/`)

라이브러리화 가능성, Docker `COPY src/` 패턴, 테스트 격리(설치된 코드 import 강제)
등을 위해 표준 src layout 유지.

```
template-dev/
├── src/project/
│   ├── config.py      # pydantic-settings 기반 설정
│   ├── exceptions.py  # 커스텀 예외 계층
│   └── utils/logging.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── ...
```

### 권장 확장 구조 (필요 시 직접 추가)

서비스/CLI 규모가 커지면 다음 모듈 분리를 고려:

- `src/project/api/` — HTTP API 진입점 (FastAPI 등)
- `src/project/cli/` — CLI 진입점
- `src/project/core/` — 도메인 로직 (외부 I/O 없음)
- `src/project/services/` — 외부 시스템 어댑터 (HTTP, DB 등)

레이어 분리를 도입한다면 의존 방향은 단방향: `api/cli → core → services`.

테스트도 작은 프로젝트는 `tests/` 하나, 외부 시스템이 엮이기 시작하면
`tests/{unit,integration}/` 분리.

### 설정 관리: pydantic-settings

`.env` + 환경 변수에서 자동으로 읽고 **타입 검증**. 누락된 필수값은 시작 시점에
즉시 `ValidationError`. (Fail Fast 원칙과 일관)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_env: Literal["development", "staging", "production"] = "development"
    database_url: str   # 필수. 누락 시 시작 못 함.

settings = Settings()
```

### Docker (Dockerfile + docker-compose.yml)

- **Multi-stage Dockerfile**: builder + runtime 분리
- **비-root 사용자** (`USER app`) — 보안
- **docker-compose**: 로컬 개발용. 보조 서비스(postgres, redis) 자리만 두고 미리 켜두지는 않음.
- 진입점(`CMD`/`ENTRYPOINT`)은 프로젝트별로 채움.

### 비동기 패턴 가이드

`CLAUDE.md` 에 async/await + 컨텍스트 매니저 + 동시성 제어 패턴 섹션 포함.
`pytest-asyncio` 설정 적용 (`asyncio_mode = "auto"`).

### CI 강화 (lint + test + typecheck)

PR마다 `make quality` (ruff) + `make typecheck` (mypy) + `make test` (pytest) 모두
통과해야 머지. mypy 는 `src/` + `tests/` 모두 strict.

---

## 양쪽 템플릿 차이 요약

| 항목 | research | dev |
|------|---------|-----|
| 레이아웃 | flat (`project/`) | src layout (`src/project/`) |
| 설정 관리 | YAML + argparse | pydantic-settings |
| 진입점 | `scripts/train.py` | 자유 (API/CLI 자체 추가) |
| Docker | ❌ | ✅ |
| 노트북 | ✅ (nbstripout 포함) | ❌ |
| 실험 추적 (W&B) | ✅ | ❌ |
| CI | lint + import sanity | lint + test + typecheck |
| mypy | `project/` 만 strict | 전체 strict |
| 비동기 패턴 가이드 | ❌ | ✅ |
| 파일 수 | ~33 | ~30 |

---

## First-time Setup Checklist (요약)

새 레포 만든 직후 한 번만:

**공통**
- [ ] README 상단 제목/설명 수정
- [ ] (선택) `pyproject.toml` 의 `name` 변경
- [ ] (선택) `project/` 디렉토리 이름 변경 (변경 시 import 경로도)
- [ ] 라이선스가 필요하면 `LICENSE` 파일 추가

**research 추가**
- [ ] `configs/default.yaml` 의 `wandb.project`, `wandb.entity` 채우기
- [ ] (필요 시) `requirements.txt` 의 PyTorch 라인 주석 해제

**dev 추가**
- [ ] `requirements.txt` 의 fastapi/uvicorn 또는 typer/click 주석 해제
- [ ] `.env.example` → `.env` 복사 후 시크릿/설정 채우기
- [ ] `Dockerfile` 의 `CMD` / `ENTRYPOINT` 를 실제 진입점으로 교체
- [ ] `docker-compose.yml` 의 ports/volumes/command 보강

---

## 보안 가이드라인 (TBD)

CLAUDE.md 의 "보안 가이드라인" 섹션은 현재 placeholder 상태입니다.

회사 보안 정책이 정해지면 다음을 포함해 채울 예정:

- 사내 보안 가이드라인 문서 링크
- 보안 영역(인증/암호화/시크릿/PII/의존성 추가) 변경 프로세스
- 취약점 리포트 채널
- AI 어시스턴트가 자동 수정하지 말아야 할 영역 (Out of Scope)

**임시 가이드** (정책 확정 전):
다음 영역은 AI 어시스턴트가 자동 수정하지 않습니다 — 변경이 필요하면 코드를
수정하지 말고 사용자에게 보고만 함.

- 인증/인가 로직
- 시크릿 관리 (`.env`, API key, 토큰)
- 외부 통신/네트워크 설정
- 개인정보(PII) 처리
- `requirements*.txt` 의존성 추가/업데이트 (보안 패치 포함)
- `.github/workflows/` 시크릿/권한 설정

---

## 알려진 한계 / 향후 개선 후보

- **템플릿 업데이트 전파**: GitHub Template은 한 번 fork되면 단절됨. 템플릿이 개선되어도 기존 레포에 자동 반영 X. (필요 시 copier 같은 도구로 마이그레이션)
- **공유 파일 drift**: 두 템플릿이 비슷한 파일을 따로 보유 (`.editorconfig`, `.gitmessage`, pre-commit, GH 템플릿 등). 한 곳 고치면 둘 다 고쳐야 함.
- **회색지대 프로젝트**: 데이터 파이프라인, ML 서비스 등 두 템플릿 사이에 걸치는 케이스. 운영 원칙은 "주된 활동" 기준으로 결정.
- **의존성 취약점 스캔 미설정**: Dependabot, `pip-audit` 등 미적용. 회사 정책 확정 후 도입.
- **템플릿 자체 CI 없음**: 템플릿 코드의 동작 검증 자동화 미구현.

---

## 자주 묻는 질문 (FAQ)

**Q. `project` 라는 이름이 placeholder 같아서 거슬려요. 바꿔야 하나요?**
A. 안 바꿔도 됩니다. 모든 사내 레포에서 `from project.utils import ...` 로 일관되는 게 오히려 장점. 정 거슬리면 First-time Setup Checklist의 "(선택)" 항목으로 변경 가능.

**Q. PyTorch가 설치 안 되어서 에러납니다.**
A. `requirements.txt` 의 torch 라인이 기본적으로 주석 처리되어 있습니다. 본인 환경의 CUDA 버전 확인 후 주석을 해제하거나, [PyTorch 공식 가이드](https://pytorch.org/get-started/locally/) 명령어로 별도 설치하세요.

**Q. conda 안 쓰고 venv 써도 되나요?**
A. 됩니다. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` 흐름도 동일하게 동작합니다. 다만 ML/DL 환경에서 CUDA 호환성 때문에 conda를 권장.
