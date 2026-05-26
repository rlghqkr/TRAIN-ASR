"""GOLD Sample → SenseVoice (FunASR) JSONL.

참고 코드: practice/2601_sensevoice_train/scripts/convert_data.py

FunASR 가 요구하는 한 줄 스키마:
    {
      "key":             "<unique>",
      "source":          "/abs/.../audio.wav",
      "source_len":      <int>,            # frames / 160 (10ms hop @ 16kHz)
      "target":          "전사 텍스트",
      "target_len":      <int>,            # 어절 수
      "text_language":   "<|ko|>",
      "emo_target":      "<|NEUTRAL|>",    # ASR 만 — NEUTRAL 고정
      "event_target":    "<|Speech|>",
      "with_or_wo_itn":  "<|woitn|>"
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from project.data.schema import Sample


# FunASR / 16kHz, hop=10ms → frames_per_sec = 100, source_len = duration_sec * 100
_FRAMES_PER_SEC_16KHZ = 100


def to_sensevoice_dict(s: Sample, *, text_field: str = "text_normalized") -> dict:
    """Sample → FunASR JSONL 한 줄 (dict)."""
    target = s.text_normalized if text_field == "text_normalized" else s.text
    return {
        "key":            s.key,
        "source":         s.audio,
        "source_len":     int(s.duration_sec * _FRAMES_PER_SEC_16KHZ),
        "target":         target,
        "target_len":     len(target.split()),
        "text_language":  "<|ko|>",
        "emo_target":     "<|NEUTRAL|>",
        "event_target":   "<|Speech|>",
        "with_or_wo_itn": "<|woitn|>",
    }


def to_sensevoice_jsonl(
    samples: Iterable[Sample],
    out_path: str | Path,
    *,
    text_field: str = "text_normalized",
) -> int:
    """GOLD samples → FunASR JSONL 파일. 반환: 줄 수."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(
                to_sensevoice_dict(s, text_field=text_field),
                ensure_ascii=False,
            ) + "\n")
            n += 1
    return n


# ─── 추론 출력 후처리 ────────────────────────────────────────────
def clean_sensevoice_output(text: str) -> str:
    """SenseVoice 출력에서 ASR 텍스트만 추출.

    원본 출력: "<|ko|><|NEUTRAL|><|Speech|>안녕하세요"  →  "안녕하세요"

    참고: funasr.utils.postprocess_utils.rich_transcription_postprocess
    가 더 정교한 처리. 여기는 간단한 fallback.
    """
    import re
    return re.sub(r"<\|[^|>]+\|>", "", text).strip()


def build_predict_fn(model_path: str, *, device: str = "cuda:0"):
    """SenseVoice 모델 로드 → predict_fn 반환 (evaluate 모듈 호환).

    Returns:
        Callable[[list[str]], list[str]] — 오디오 경로 리스트 → 텍스트 리스트.
    """
    try:
        from funasr import AutoModel
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess as _post
        except ImportError:
            _post = clean_sensevoice_output
    except ImportError as e:
        raise ImportError(
            "funasr 가 필요합니다. `pip install funasr`"
        ) from e

    model = AutoModel(model=model_path, trust_remote_code=True, device=device)

    def predict(audio_paths: list[str]) -> list[str]:
        results = model.generate(
            input=audio_paths,
            language="auto",
            use_itn=False,
            batch_size_s=300,
        )
        return [_post(r["text"]) for r in results]

    return predict
