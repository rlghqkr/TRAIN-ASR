# Project Name

> Research project — (한 줄 요약)

## First-time Setup Checklist

이 템플릿으로 새 레포를 만든 직후 한 번만:

- [ ] 이 README 상단 제목/설명 수정
- [ ] (선택) `pyproject.toml` 의 `name` 변경
- [ ] (선택) `project/` 디렉토리 이름 변경 (변경 시 `from project.utils ...` import 경로도 함께 수정)
- [ ] `configs/default.yaml` 의 `wandb.project`, `wandb.entity` 채우기
- [ ] 라이선스가 필요하면 `LICENSE` 파일 추가
- [ ] 이 체크리스트 섹션 삭제

## Quick Start

### 1. 환경 생성 (conda)

```bash
conda create -n <env-name> python=3.11 -y
conda activate <env-name>
```

### 2. 의존성 설치

```bash
make setup
```

이 명령은 다음을 수행합니다:
- `requirements.txt` + `requirements-dev.txt` 설치
- `pre-commit` 훅 설치
- git commit 메시지 템플릿 설정

> **PyTorch 설치**: 서버/머신마다 CUDA 버전이 다르므로 `requirements.txt` 의
> torch 라인은 기본적으로 **주석 처리**되어 있습니다. 본인 환경의 CUDA 버전에 맞춰
> 주석을 해제하거나, [PyTorch 공식 가이드](https://pytorch.org/get-started/locally/) 명령어로 별도 설치하세요.

### 3. 환경 변수

```bash
cp .env.example .env
# .env 편집 — WANDB_API_KEY 등
```

### 4. 학습 실행

```bash
python scripts/train.py --config configs/default.yaml
# 또는
make train CFG=configs/default.yaml
```

## Project Structure

기본 골격은 최소화되어 있습니다. 프로젝트가 커지면 자유롭게 확장하세요.

```
.
├── configs/         # 실험 설정 (YAML)
├── data/            # 데이터 (커밋 안 됨)
├── notebooks/       # 탐색/분석 노트북
├── reports/         # 그림/리포트
├── scripts/         # 진입점 (train.py 등)
├── project/
│   └── utils/       # 재현성(seed), config 로더, W&B 헬퍼
└── tests/
```

권장 확장 패턴(예: `project/{data,models,training,evaluation,viz}/`)은
[`CLAUDE.md`](./CLAUDE.md)의 "프로젝트 구조" 섹션을 참고하세요.

## Development

```bash
make style       # 포맷 + 린트 자동 수정
make quality     # 포맷/린트 검사
make test        # 테스트 실행
make typecheck   # mypy
make clean       # 캐시 제거
```

## Experiments

새 실험 추가:

1. `configs/<exp_name>.yaml` 작성 (default.yaml 기반)
2. `python scripts/train.py --config configs/<exp_name>.yaml` 실행
3. W&B run link을 PR/이슈에 기록
4. `exp: ...` 형식으로 commit
