# =============================================================================
# template-research Makefile
# =============================================================================
# 의존성은 `make setup` 한 번에 설치되며, 이후 명령어는 재설치 없이 동작합니다.

.PHONY: help setup style quality test typecheck train clean clean-pyc clean-test

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  setup       - 의존성 + pre-commit 설치 (최초 1회)"
	@echo "  style       - ruff로 포맷 + lint 자동 수정"
	@echo "  quality     - ruff로 포맷/lint 검사 (CI에서 사용)"
	@echo "  test        - pytest 실행"
	@echo "  typecheck   - mypy 타입 검사"
	@echo "  train       - 학습 실행 (CFG=configs/default.yaml)"
	@echo "  docs-study  - docs/study/index.html (단일 인터랙티브 HTML) 재빌드"
	@echo "  clean       - 캐시/임시 파일 제거"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	git config --local commit.template .gitmessage
	@echo ""
	@echo "✅ Setup complete."
	@echo "   - PyTorch는 환경에 맞춰 requirements.txt 의 torch 라인을 주석 해제 후 별도 설치하세요."
	@echo "   - .env.example 을 .env 로 복사해서 시크릿을 채우세요."

# -----------------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------------
style:
	ruff check --fix .
	ruff format .

quality:
	ruff check .
	ruff format --check .

# -----------------------------------------------------------------------------
# Test / Typecheck
# -----------------------------------------------------------------------------
test:
	python -m pytest tests/

typecheck:
	python -m mypy

# -----------------------------------------------------------------------------
# Research
# -----------------------------------------------------------------------------
CFG ?= configs/default.yaml

train:
	python scripts/train.py --config $(CFG)

# -----------------------------------------------------------------------------
# Docs
# -----------------------------------------------------------------------------
docs-study:
	python scripts/build_study_html.py

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
clean: clean-pyc clean-test

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
