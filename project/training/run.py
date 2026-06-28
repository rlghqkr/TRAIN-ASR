"""학습 오케스트레이션 — CLI(scripts/train.py)·노트북 공통 진입점.

가이드 GUIDELINE/2_모델학습.md 큰그림의 ①②③ 흐름을 한 함수로 묶는다:

    load_samples → validate → 모델별 어댑터 변환 → build_trainer → train → 저장

학습 로직은 여기 한 곳에만 둔다(SoT). CLI 도 노트북도 이 함수를 호출만 한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import yaml

from project.data import load_samples


logger = structlog.get_logger()


def run_whisper_training(
    cfg: dict[str, Any],
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Whisper 파인튜닝 한 번을 끝까지 수행.

    Args:
        cfg: load_config() 로 읽고 보간 해석된 설정 dict.
        run_name: W&B run 이름 override (미지정 시 experiment.name 사용).

    Returns:
        {"output_dir": str, "log_history": list} — 저장 위치와 학습 로그.

    Raises:
        SchemaValidationError: train/val JSONL 검증 실패 시 (Fail Fast).
    """
    from project.data.adapters.whisper import to_whisper_dataset
    from project.training.whisper import build_trainer

    exp_name = cfg["experiment"]["name"]
    log = logger.bind(experiment=exp_name, model="whisper")

    m = cfg["models"]["whisper"]
    backbone = m["backbone"]
    language = m.get("language", "ko")
    task = m.get("task", "transcribe")

    data_cfg = cfg["data"]
    text_field = data_cfg["text_field"]
    sample_rate = data_cfg["sample_rate"]

    paths = cfg["paths"]
    train_jsonl = paths["train_jsonl"]
    val_jsonl = paths["val_jsonl"]

    if run_name:
        os.environ.setdefault("WANDB_NAME", run_name)

    # ── 1) 데이터 로드 + 검증 ────────────────────────────────────────
    # 학습/평가가 같은 스키마를 공유 (project/data/schema.py).
    # 스키마 위반 행은 죽지 말고 건너뛰고 경고(skip_invalid) — 수십만 건 중 몇 줄 때문에
    # 긴 학습이 로드 단계에서 터지지 않게. (조용한 누락 아님: 건수/사유 경고 로그 남김)
    train_samples = load_samples(train_jsonl, skip_invalid=True)
    val_samples = load_samples(val_jsonl, skip_invalid=True)

    # 학습 데이터 부분 샘플링 (선택) — data.train_sample_frac < 1.0 이면 학습셋에서
    # 시드(experiment.seed) 고정 무작위 부분집합만 학습한다. 큰 학습셋에서 빠른 실험/디버그용.
    # 같은 시드면 항상 같은 부분집합(재현성). val 은 작아서 보통 전량 그대로 둔다.
    train_frac = float(data_cfg.get("train_sample_frac", 1.0))
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"data.train_sample_frac 는 (0, 1] 범위여야 함: {train_frac}")
    if train_frac < 1.0:
        import random
        seed = int(cfg["experiment"].get("seed", 42))
        n_total = len(train_samples)
        n_keep = max(1, round(n_total * train_frac))
        idx = sorted(random.Random(seed).sample(range(n_total), n_keep))
        train_samples = [train_samples[i] for i in idx]
        log.info("Subsampled train data", frac=train_frac,
                 n_kept=len(train_samples), n_total=n_total)

    log.info("Loaded samples", n_train=len(train_samples), n_val=len(val_samples))

    # ── 2) 모델별 어댑터 → HuggingFace Dataset ───────────────────────
    train_ds = to_whisper_dataset(
        train_samples,
        backbone=backbone,
        language=language,
        task=task,
        text_field=text_field,
        sampling_rate=sample_rate,
    )
    val_ds = to_whisper_dataset(
        val_samples,
        backbone=backbone,
        language=language,
        task=task,
        text_field=text_field,
        sampling_rate=sample_rate,
    )

    # ── 3) Trainer 빌드 ──────────────────────────────────────────────
    trainer = build_trainer(cfg, train_dataset=train_ds, eval_dataset=val_ds)

    # ── 4) 학습 (중단 후 재개는 training.resume_from 로) ──────────────
    resume = cfg["training"].get("resume_from")
    log.info("Training started", resume_from=resume)
    trainer.train(resume_from_checkpoint=resume)
    log.info("Training finished")

    # ── 5) 모델 + 재현용 config 저장 ─────────────────────────────────
    output_dir = Path(paths["outputs_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Saved model + config", output_dir=str(output_dir))

    log_history = trainer.state.log_history

    # ── 6) GPU 메모리 해제 ───────────────────────────────────────────
    # 노트북에서 호출 시 커널이 학습 후에도 GPU 를 계속 점유하지 않도록
    # trainer(모델·옵티마이저 상태) 참조를 끊고 CUDA 캐시를 비운다.
    import gc

    import torch

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Released GPU memory")

    return {
        "output_dir": str(output_dir),
        "log_history": log_history,
    }
