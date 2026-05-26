"""CER / sCER / WER 계산 + 슬라이스 분해.

표준: jiwer corpus-level (총 거리 / 총 문자). 발화당 CER 도 함께 제공 (분석용).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .normalize import normalize_korean_asr


# ───────────────────────────────────────────────────────────────────────
# jiwer 가 없으면 친절한 에러
# ───────────────────────────────────────────────────────────────────────
def _require_jiwer():
    try:
        import jiwer
        return jiwer
    except ImportError as e:
        raise ImportError(
            "jiwer 가 필요합니다. `pip install jiwer` 후 다시 실행하세요."
        ) from e


# ───────────────────────────────────────────────────────────────────────
# 결과 컨테이너
# ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CerResult:
    cer: float          # %, corpus-level
    scer: float         # %, 공백 무시
    wer: float          # %, 참고
    samples: int
    # 발화당 (분석용)
    per_sample_cer: list[float] = field(default_factory=list)


# ───────────────────────────────────────────────────────────────────────
# 핵심 계산
# ───────────────────────────────────────────────────────────────────────
def compute_cer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalize: bool = True,
) -> CerResult:
    """corpus-level CER / sCER / WER.

    Args:
        references: 정답 텍스트.
        hypotheses: 모델 출력 텍스트.
        normalize: True 면 양쪽 모두 한국어 정규화 후 비교.

    Returns:
        CerResult — % 단위.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"refs ({len(references)}) != hyps ({len(hypotheses)})"
        )

    jiwer = _require_jiwer()

    if normalize:
        refs = [normalize_korean_asr(r) for r in references]
        hyps = [normalize_korean_asr(h) for h in hypotheses]
    else:
        refs, hyps = list(references), list(hypotheses)

    # 빈 reference 는 jiwer 가 ZeroDivision 일으킴 — 미리 필터
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return CerResult(cer=0.0, scer=0.0, wer=0.0, samples=0)
    refs_f, hyps_f = zip(*pairs)

    cer_val = jiwer.cer(list(refs_f), list(hyps_f)) * 100
    wer_val = jiwer.wer(list(refs_f), list(hyps_f)) * 100

    # sCER: 공백 무시
    refs_ns = [r.replace(" ", "") for r in refs_f]
    hyps_ns = [h.replace(" ", "") for h in hyps_f]
    scer_val = jiwer.cer(refs_ns, hyps_ns) * 100

    # 발화당 (분석용)
    per_sample: list[float] = []
    for r, h in pairs:
        try:
            per_sample.append(jiwer.cer(r, h) * 100)
        except Exception:
            per_sample.append(float("nan"))

    return CerResult(
        cer=cer_val, scer=scer_val, wer=wer_val,
        samples=len(pairs), per_sample_cer=per_sample,
    )


# ───────────────────────────────────────────────────────────────────────
# 슬라이스 평가
# ───────────────────────────────────────────────────────────────────────
def slice_cer(
    samples: Sequence[dict],
    *,
    ref_field: str = "text_normalized",
    hyp_field: str = "prediction_normalized",
    slice_field: str,
    normalize: bool = False,    # 이미 *_normalized 라면 False
) -> dict[str, CerResult]:
    """슬라이스(메타 필드 값별) CER.

    Args:
        samples: dict 리스트. 각 dict 는 ref_field, hyp_field, slice_field 보유.
        slice_field: 'age_group', 'gender', 'corpus_id' 등.

    Returns:
        {slice_value: CerResult}
    """
    buckets: dict[str, list[tuple[str, str]]] = {}
    for s in samples:
        v = s.get(slice_field, "unknown") or "unknown"
        buckets.setdefault(str(v), []).append(
            (s.get(ref_field, ""), s.get(hyp_field, ""))
        )

    out: dict[str, CerResult] = {}
    for k, pairs in buckets.items():
        refs, hyps = zip(*pairs) if pairs else ([], [])
        out[k] = compute_cer(refs, hyps, normalize=normalize)
    return out


# ───────────────────────────────────────────────────────────────────────
# bootstrap 신뢰구간 (작은 평가셋용)
# ───────────────────────────────────────────────────────────────────────
def bootstrap_cer_ci(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    n_iter: int = 1000,
    seed: int = 42,
    normalize: bool = True,
) -> tuple[float, float]:
    """95% bootstrap 신뢰구간 (CER %). 작은 벤치마크용."""
    import random
    rng = random.Random(seed)

    pairs = list(zip(references, hypotheses))
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)

    scores: list[float] = []
    for _ in range(n_iter):
        sample = [rng.choice(pairs) for _ in range(n)]
        refs, hyps = zip(*sample)
        scores.append(compute_cer(refs, hyps, normalize=normalize).cer)

    scores.sort()
    lo = scores[int(n_iter * 0.025)]
    hi = scores[int(n_iter * 0.975)]
    return (lo, hi)
