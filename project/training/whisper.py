"""Whisper 학습 래퍼 — HuggingFace Trainer 기반.

JupyterLab 셀 내 `trainer.train()` 직접 실행 가능 (SenseVoice 와 차이).
장시간 학습은 그래도 tmux 권장.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _require_deps():
    try:
        import transformers   # noqa: F401
        import accelerate     # noqa: F401
    except ImportError as e:
        raise ImportError(
            "transformers / accelerate 필요. "
            "`pip install transformers accelerate`"
        ) from e


def build_trainer(
    cfg: dict[str, Any],
    *,
    train_dataset,
    eval_dataset,
):
    """config + 어댑터 통과 datasets → HuggingFace Trainer.

    Args:
        cfg: configs/<exp>.yaml dict.
        train_dataset / eval_dataset: `to_whisper_dataset()` 산출.

    Returns:
        transformers.Seq2SeqTrainer
    """
    _require_deps()
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    m = cfg["models"]["whisper"]
    backbone = m["backbone"]
    language = m.get("language", "ko")
    task = m.get("task", "transcribe")

    processor = WhisperProcessor.from_pretrained(backbone, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(backbone)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None    # decoder prompt 는 별도 설정

    tr = cfg["training"]
    output_dir = Path(cfg["paths"]["outputs_dir"]) / cfg["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_steps = tr.get("eval_every", 1000)
    save_steps = tr.get("save_every", 1000)

    # early stopping 을 쓰려면 best 모델 추적이 필요하고, 그 전제로
    # save_steps 가 eval_steps 의 배수여야 한다(HF 제약). patience 설정 시 정렬.
    patience = tr.get("early_stopping_patience")
    load_best = patience is not None
    if load_best and save_steps % eval_steps != 0:
        save_steps = eval_steps

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=tr["batch_size"],
        per_device_eval_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr.get("grad_accum_steps", 1),
        learning_rate=tr["lr"],
        lr_scheduler_type=tr.get("scheduler", "linear"),   # cosine | linear | constant
        warmup_steps=tr.get("warmup_steps", 0),
        weight_decay=tr.get("weight_decay", 0.0),
        max_grad_norm=tr.get("grad_clip", 1.0),
        num_train_epochs=tr["epochs"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=tr.get("log_every", 25),
        dataloader_num_workers=cfg.get("data", {}).get("num_workers", 0),
        bf16=(tr.get("precision") == "bf16"),
        fp16=(tr.get("precision") == "fp16"),
        predict_with_generate=True,
        generation_max_length=cfg.get("data", {}).get("max_label_len", 200),
        # CER 로 best 모델 선택 (낮을수록 좋음 → greater_is_better=False)
        load_best_model_at_end=load_best,
        metric_for_best_model="cer" if load_best else None,
        greater_is_better=False if load_best else None,
        report_to=["wandb"] if cfg.get("wandb", {}).get("enabled") else [],
        run_name=cfg["experiment"]["name"],
    )

    # W&B 환경 변수
    if cfg.get("wandb", {}).get("enabled"):
        os.environ.setdefault("WANDB_PROJECT",
                              cfg["wandb"].get("project", "train-asr"))
        if cfg["wandb"].get("group"):
            os.environ.setdefault("WANDB_RUN_GROUP", cfg["wandb"]["group"])

    data_collator = _build_collator(processor)

    callbacks = []
    if patience is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,    # transformers 5.x: tokenizer 인자 대체. 전체 processor 를 넘겨 저장 시 tokenizer 까지 포함.
        compute_metrics=_build_compute_metrics(processor),
        callbacks=callbacks,
    )
    return trainer


def _build_compute_metrics(processor):
    """eval 스텝마다 CER/sCER/WER 를 계산하는 콜백 (loss 만 보지 않도록).

    기존 평가 모듈(project.evaluation.compute_cer)을 그대로 재사용한다.
    """
    import numpy as np

    from project.evaluation import compute_cer

    tokenizer = processor.tokenizer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # collator 가 -100 으로 마스킹한 패딩을 pad_token 으로 되돌려 디코딩
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 학습/평가 일관되게 한국어 정규화 후 비교 (CER %)
        res = compute_cer(label_str, pred_str, normalize=True)
        return {"cer": res.cer, "scer": res.scer, "wer": res.wer}

    return compute_metrics


def _build_collator(processor):
    """Whisper 학습용 data collator (input_features padding + label -100 masking)."""
    import torch

    class _Collator:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features: list[dict]):
            input_features = [
                {"input_features": f["input_features"]} for f in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            # decoder_start_token 이 labels 의 첫 토큰이면 제거 (Whisper 규약)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    return _Collator(processor)
