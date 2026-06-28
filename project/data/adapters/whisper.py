"""GOLD Sample → Whisper (HuggingFace) Dataset.

학습/평가 양쪽에서 사용. 핵심 변환:
    {audio: '/abs/.../wav', text_norm: '...'}  →
    {input_features: [...], labels: [token_ids]}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from project.data.schema import Sample

# 어댑터는 import 가벼움 유지(structlog 미사용) → 경고는 stdlib logging 으로.
_log = logging.getLogger(__name__)


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


def _load_audio(path: str, target_sr: int, *, on_error: str = "raise"):
    """오디오 로드 + 필요 시 target_sr 로 리샘플.

    on_error 로 실패(파일 누락/깨짐) 시 동작을 정한다:
    - "raise"  (기본): 예외 그대로 — 학습 데이터 무결성은 Fail-Fast.
    - "silence": 경고 후 0.1s 무음 반환 — 평가용(배치 길이 정렬 유지, 그 발화는 빈/오답 처리).
    - "skip"   : 경고 후 None 반환 — 학습용(호출부가 그 발화를 건너뜀. 무음을 학습하지 않게).

    원본 SR 이 다르면(예: 8kHz 전화망) **로드 시점에** 리샘플한다 — 미리 16k 로 변환·저장하지
    않기 위함(SILVER 가 RAW 를 참조(ref)할 수 있게). 리샘플 비용(~1.6ms/발화)은 mel 피처추출
    (~3.9ms)보다 작고 데이터로더 워커에서 병렬·GPU 와 오버랩되어 병목이 아님.

    KsponSpeech `.pcm` 은 **헤더 없는 raw PCM(16kHz/mono/s16le)** → 로드 시점에 헤더를 가상
    적용해 읽는다(변환·저장 없이 원본 참조). 그 외(wav/flac)는 soundfile 로 읽는다.
    """
    import numpy as np

    try:
        if str(path).lower().endswith(".pcm"):
            # raw PCM 16kHz mono int16 → float32 [-1,1]
            audio = np.frombuffer(open(path, "rb").read(), dtype="<i2").astype(np.float32) / 32768.0
            sr = 16000
        else:
            import soundfile as sf

            audio, sr = sf.read(path)
        if sr != target_sr:
            import torch
            import torchaudio

            audio = torchaudio.functional.resample(
                torch.as_tensor(audio, dtype=torch.float32), sr, target_sr
            ).numpy()
        return audio
    except Exception as e:
        if on_error == "raise":
            raise
        # 조용한 누락이 아니라 어느 파일이 왜 실패했는지 경고로 남긴다.
        _log.warning("오디오 로드 실패 — %s: %s (%s: %s)",
                     "무음 대체(평가 계속)" if on_error == "silence" else "발화 건너뜀(학습)",
                     path, type(e).__name__, e)
        if on_error == "silence":
            return np.zeros(target_sr // 10, dtype=np.float32)  # 0.1s 무음
        return None  # on_error == "skip" → 호출부가 이 발화를 건너뜀


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
            # 학습은 깨진 오디오를 무음으로 때우면 안 됨 → 그 발화 자체를 건너뜀(경고는 _load_audio 가 남김).
            audio = _load_audio(s.audio, sampling_rate, on_error="skip")
            if audio is None:
                continue
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
    backbone: str | None = None,
    language: str = "ko",
    task: str = "transcribe",
    device: str = "cuda:0",
    batch_size: int = 16,
    beam_size: int = 5,
):
    """Whisper 모델 로드 → predict_fn 반환 (evaluate 모듈 호환).

    `model_path` 만으로 평가하는 게 기본이다. 학습 산출물(trainer.save_model)이든
    원본 백본이든 HF 폴더는 모델·프로세서를 모두 담고 있으므로 폴더 하나면 충분하다.
    `backbone` 은 가중치만 든 `.pt` 파일을 평가할 때(프로세서 출처가 없을 때)만 필요.

    Args:
        model_path: 평가할 모델. HF 폴더(원본 백본 or 학습 체크포인트) 또는 `.pt` 가중치 파일.
        backbone: `.pt` 가중치 평가 시 프로세서/기본 아키텍처 출처. 폴더 평가 땐 불필요(None).
        language / task: forced decoder language / task.
        device: torch device. 런처(scripts/eval.sh)가 CUDA_VISIBLE_DEVICES 로 물리 GPU 를
                마스킹하므로 보통 논리 'cuda:0' 그대로 둔다.

    Returns:
        Callable[[list[str]], list[str]] — 오디오 경로 → 텍스트.

    Raises:
        FileNotFoundError: model_path 가 폴더도 파일도 아닐 때.
        ValueError: `.pt` 가중치인데 backbone 이 없을 때 (프로세서 출처 부재).
    """
    _require_infer_deps()
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_path = Path(model_path)
    if model_path.is_dir():
        # self-contained HF 폴더 — 모델·프로세서 모두 여기서 로드 (백본/체크포인트 공통)
        processor = WhisperProcessor.from_pretrained(model_path, language=language, task=task)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    elif model_path.is_file():
        # 가중치만 든 .pt — 프로세서/기본 아키텍처는 backbone 에서 가져와야 함
        if not backbone:
            raise ValueError(
                f".pt 가중치 평가에는 backbone 이 필요합니다 (프로세서 출처). model_path={model_path}"
            )
        processor = WhisperProcessor.from_pretrained(backbone, language=language, task=task)
        model = WhisperForConditionalGeneration.from_pretrained(backbone)
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:  # {"model": state_dict} 래핑 처리
            state = state["model"]
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"model_path 가 폴더도 파일도 아님: {model_path}")

    model.eval().to(device)

    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    def predict(audio_paths: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i:i + batch_size]
            # 평가는 배치 길이 정렬이 필요 → 깨진 오디오는 무음 대체(그 발화만 빈/오답). 안 죽음.
            audios = [_load_audio(p, 16000, on_error="silence") for p in batch]
            inputs = processor(
                audios, sampling_rate=16000, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                # 입력 mel 을 모델 dtype 에 맞춤. large-v3 처럼 가중치가 fp16 인 폴더는
                # processor 출력(fp32)과 dtype 이 달라 conv 에서 float/Half 충돌이 난다.
                # fp32 모델이면 no-op.
                ids = model.generate(
                    inputs.input_features.to(model.dtype),
                    forced_decoder_ids=forced_ids,
                    num_beams=beam_size,
                    max_new_tokens=200,
                )
            decoded = processor.batch_decode(ids, skip_special_tokens=True)
            out.extend(clean_whisper_output(d) for d in decoded)
        return out

    return predict
