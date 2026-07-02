"""GOLD Sample → Whisper (HuggingFace) Dataset.

학습/평가 양쪽에서 사용. 핵심 변환:
    {audio: '/abs/.../wav', text_norm: '...'}  →
    {input_features: [...], labels: [token_ids]}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Literal

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


def _whisper_feature_gen(
    samples: list[Sample],
    *,
    backbone: str,
    language: str,
    task: str,
    text_field: str,
    sampling_rate: int,
):
    """log-mel 피처 + label 토큰을 산출하는 제너레이터 (사전계산용).

    `Dataset.from_generator(..., num_proc=N)` 로 호출되면 datasets 가 `samples`
    리스트를 N 개 샤드로 쪼개 각 워커 프로세스에서 이 함수를 돈다. 그래서
    feature_extractor / tokenizer 는 **프로세스마다 새로 로드**한다(피클 회피 + 프로세스
    독립). 모듈 최상위 함수라 datasets 가 안정적으로 fingerprint(캐시 키)를 계산할 수
    있어, 같은 입력이면 캐시를 재사용한다.

    깨진/누락 오디오는 학습에 무음을 섞지 않도록 그 발화를 건너뛴다
    (경고는 _load_audio 가 남김).
    """
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    feature_extractor = WhisperFeatureExtractor.from_pretrained(backbone)
    tokenizer = WhisperTokenizer.from_pretrained(backbone, language=language, task=task)

    for s in samples:
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


def _on_the_fly_dataset(
    samples: list[Sample],
    *,
    backbone: str,
    language: str,
    task: str,
    text_field: str,
    sampling_rate: int,
):
    """메타데이터(오디오 경로 + 텍스트)만 가진 Dataset + `set_transform` — **on-the-fly** 모드.

    피처를 디스크에 미리 굽지 않고 **collate 시점에 그때그때 추출**한다(DataLoader 워커가 GPU 와
    overlap). precompute 와 학습 결과는 동일(같은 결정적 피처)하되:
      - 디스크 저장 0 (precompute 의 ~600GB/30초패딩 낭비 없음), 느린 HDD 사전계산 단계 없음.
      - 단 **매 epoch 피처를 재계산**(1 epoch 이면 총 계산량 precompute 와 동일). 워커가 GPU 와
        겹쳐 학습속도 영향은 보통 작다(`data.num_workers` 8~16 권장).

    주의(깨진 오디오): `set_transform` 은 row 를 못 버린다 → 깨진/누락 오디오는 **무음 대체(경고
    로그)**. precompute 는 그 발화를 건너뛰지만 여기선 불가. SILVER 가 음원 존재를 검증하므로 깨진
    건 드물다. 엄격히 배제하려면 매니페스트를 사전 필터링(별도).
    """
    import datasets as hfds
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    feature_extractor = WhisperFeatureExtractor.from_pretrained(backbone)
    tokenizer = WhisperTokenizer.from_pretrained(backbone, language=language, task=task)

    ds = hfds.Dataset.from_dict({
        "id": [s.key for s in samples],
        "audio": [s.audio for s in samples],
        "text": [(s.text_norm if text_field == "text_norm" else s.text) for s in samples],
    })

    def _transform(batch: dict) -> dict:
        feats, labs = [], []
        for path, text in zip(batch["audio"], batch["text"]):
            audio = _load_audio(path, sampling_rate, on_error="silence")  # 깨지면 무음(경고 로그)
            feats.append(
                feature_extractor(
                    audio, sampling_rate=sampling_rate, return_tensors="np",
                ).input_features[0]
            )
            labs.append(tokenizer(text, return_tensors="np").input_ids[0])
        return {"id": batch["id"], "input_features": feats, "labels": labs}

    ds.set_transform(_transform)
    return ds


def to_whisper_dataset(
    samples: Iterable[Sample],
    *,
    backbone: str = "openai/whisper-small",
    language: str = "ko",
    task: str = "transcribe",
    text_field: str = "text_norm",
    sampling_rate: int = 16000,
    cache_dir: str | Path | None = None,
    num_proc: int = 1,
    feature_mode: Literal["precompute", "on_the_fly"] = "precompute",
):
    """GOLD samples → HuggingFace Dataset (input_features + labels).

    **모드 2종** (`feature_mode`, 학습 결과는 동일 — 같은 결정적 피처):
      - `"precompute"`(기본): log-mel 을 **미리 다 추출해 Arrow 캐시로 굽는다**. 한 번 굽고 N epoch
        재사용(계산 1회). 단 ~0.96MB/샘플 × 수십만 = 수백 GB 디스크 + 사전계산 시간(HDD 병목).
        대용량은 `cache_dir`(대용량 디스크)·`num_proc`(병렬) 필수.
      - `"on_the_fly"`: 미리 안 굽고 **학습 중 collate 시점에 추출**(`set_transform`). 디스크 0,
        30초패딩 저장 낭비 없음. 매 epoch 재계산되나 DataLoader 워커가 GPU 와 overlap → 속도 영향
        보통 작음(`data.num_workers` 8~16 권장). 대용량/디스크 부족에 유리. (깨진 오디오 처리 차이는
        `_on_the_fly_dataset` 주석 참조.)

    아래 설명은 `"precompute"` 기준. 수십만 건 풀셋은 피처가
    ~0.96MB/샘플(80×3000 float32)이라 통째로 수백 GB 가 되므로:
      - `cache_dir` 는 대용량 디스크를 가리켜야 한다(홈/`/` 는 못 담아 `No space left` 로 터짐).
      - `num_proc` 로 병렬 추출한다(단일 프로세스는 수십만 건에 수 시간).
    같은 samples + 인자면 fingerprint 가 같아 캐시를 재사용한다(재실행 시 재계산 안 함).

    Args:
        samples: GOLD Sample 리스트.
        backbone: Whisper 백본 (HF ID or 로컬 경로).
        language / task: forced decoder language / task.
        text_field: 'text' 또는 'text_norm'.
        sampling_rate: 16000 권장.
        cache_dir: 피처 Arrow 캐시 위치. None 이면 datasets 기본(홈 캐시) — 풀셋엔 반드시
            대용량 디스크 경로를 줄 것.
        num_proc: 사전계산 병렬 프로세스 수. samples 수보다 크면 자동 보정.

    Returns:
        datasets.Dataset
    """
    _require_train_deps()

    import datasets as hfds

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("samples 가 비어 있음")

    if feature_mode == "on_the_fly":
        return _on_the_fly_dataset(
            sample_list, backbone=backbone, language=language, task=task,
            text_field=text_field, sampling_rate=sampling_rate,
        )
    if feature_mode != "precompute":
        raise ValueError(
            f"feature_mode 는 'precompute' | 'on_the_fly' 여야 함 (받음: {feature_mode!r})"
        )

    # num_proc 가 샘플 수보다 크면 빈 샤드가 생겨 datasets 가 에러 → 상한 보정.
    n_proc = max(1, min(int(num_proc), len(sample_list)))

    return hfds.Dataset.from_generator(
        _whisper_feature_gen,
        gen_kwargs={
            "samples": sample_list,          # list → datasets 가 num_proc 샤드로 분할
            "backbone": backbone,
            "language": language,
            "task": task,
            "text_field": text_field,
            "sampling_rate": sampling_rate,
        },
        num_proc=n_proc,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


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
    # 평가도 학습부(build_trainer)와 동일하게 generate(language=, task=) 로 디코더 프롬프트를 강제한다.
    # 구 forced_decoder_ids 방식은 deprecated 이고 학습 설정과 불일치 → 언어/태스크 강제가 조용히
    # 새면 CER 이 실제보다 나쁘게 나온다. 모델에 박힌 forced_decoder_ids 는 비워 충돌 방지.
    model.generation_config.forced_decoder_ids = None

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
                    language=language,
                    task=task,
                    num_beams=beam_size,
                    max_new_tokens=200,
                )
            decoded = processor.batch_decode(ids, skip_special_tokens=True)
            out.extend(clean_whisper_output(d) for d in decoded)
        return out

    return predict
