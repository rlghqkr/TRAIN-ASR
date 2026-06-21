"""GOLD Sample → Whisper (HuggingFace) Dataset.

학습/평가 양쪽에서 사용. 핵심 변환:
    {audio: '/abs/.../wav', text_norm: '...'}  →
    {input_features: [...], labels: [token_ids]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from project.data.schema import Sample


def _require_train_deps():
    """학습 (to_whisper_dataset) 용 — datasets 까지 필요."""
    try:
        import transformers   # noqa: F401
        import datasets       # noqa: F401
        import soundfile      # noqa: F401
    except ImportError as e:
        raise ImportError(
            "transformers / datasets / soundfile 필요. "
            "`pip install transformers datasets soundfile`"
        ) from e


def _require_infer_deps():
    """추론 (build_predict_fn) 용 — datasets 불필요."""
    try:
        import transformers   # noqa: F401
        import soundfile      # noqa: F401
    except ImportError as e:
        raise ImportError(
            "transformers / soundfile 필요. `pip install transformers soundfile`"
        ) from e


def to_whisper_dataset(
    samples: Iterable[Sample],
    *,
    backbone: str = "openai/whisper-small",
    language: str = "ko",
    task: str = "transcribe",
    text_field: str = "text_norm",
    sampling_rate: int = 16000,
    cache_dir: str | Path | None = None,
):
    """GOLD samples → HuggingFace Dataset (input_features + labels).

    Args:
        samples: GOLD Sample 리스트.
        backbone: Whisper 백본 (HF ID or 로컬 경로).
        language / task: forced decoder language / task.
        text_field: 'text' 또는 'text_norm'.
        sampling_rate: 16000 권장.

    Returns:
        datasets.Dataset
    """
    _require_train_deps()

    import datasets as hfds
    import soundfile as sf
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("samples 가 비어 있음")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(backbone)
    tokenizer = WhisperTokenizer.from_pretrained(
        backbone, language=language, task=task,
    )

    def gen():
        for s in sample_list:
            audio, sr = sf.read(s.audio)
            if sr != sampling_rate:
                raise ValueError(
                    f"sample_rate mismatch: {s.audio} = {sr}, expected {sampling_rate}"
                )
            text = s.text_norm if text_field == "text_norm" else s.text
            features = feature_extractor(
                audio, sampling_rate=sampling_rate, return_tensors="np",
            ).input_features[0]
            labels = tokenizer(text, return_tensors="np").input_ids[0]
            yield {
                "id": s.key,
                "input_features": features,
                "labels": labels,
            }

    return hfds.Dataset.from_generator(gen)


# ─── 추론 출력 후처리 ────────────────────────────────────────────
def clean_whisper_output(text: str) -> str:
    """Whisper 출력 정리. transformers 의 `skip_special_tokens=True` 면 대부분 제거됨.

    혹시 남아 있을 `<|...|>` 토큰 제거.
    """
    import re
    return re.sub(r"<\|[^|>]+\|>", "", text).strip()


def build_predict_fn(
    model_path: str | Path,
    *,
    backbone: str = "openai/whisper-small",
    language: str = "ko",
    task: str = "transcribe",
    device: str = "cuda:0",
    batch_size: int = 16,
    beam_size: int = 5,
):
    """Whisper 모델 로드 → predict_fn 반환 (evaluate 모듈 호환).

    Returns:
        Callable[[list[str]], list[str]] — 오디오 경로 → 텍스트.
    """
    _require_infer_deps()
    import torch
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(backbone, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(backbone)

    model_path = Path(model_path)
    if model_path.is_file():
        # 체크포인트 가중치 적용
        state = torch.load(model_path, map_location="cpu")
        # state_dict 가 dict 안에 또 dict 인 경우도 처리
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    elif model_path.is_dir():
        try:
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
        except Exception:
            pass

    model.eval().to(device)

    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    def predict(audio_paths: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i:i + batch_size]
            audios = []
            for p in batch:
                a, sr = sf.read(p)
                if sr != 16000:
                    raise ValueError(f"sample_rate != 16000: {p}")
                audios.append(a)
            inputs = processor(
                audios, sampling_rate=16000, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                ids = model.generate(
                    inputs.input_features,
                    forced_decoder_ids=forced_ids,
                    num_beams=beam_size,
                    max_new_tokens=200,
                )
            decoded = processor.batch_decode(ids, skip_special_tokens=True)
            out.extend(clean_whisper_output(d) for d in decoded)
        return out

    return predict
