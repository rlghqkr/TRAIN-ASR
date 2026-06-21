"""한국어 ASR 전사 정규화 — 한 곳에서 관리되는 규칙.

이 함수는 *두 곳* 에서 동일하게 호출된다:
1. 데이터 빌드 시 text_norm 채우기
2. 평가 시 reference / hypothesis 양쪽 모두

규칙 변경은 **모든 모델의 CER 을 변동시킨다**. PR + 영향 분석 필수.

자세한 정책: docs/evaluation/metrics.md / docs/data/schema.md §8
참고: https://github.com/rtzr/Awesome-Korean-Speech-Recognition (이중 전사 / 정규화 이슈)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal


# ───────────────────────────────────────────────────────────────────────
# 모델 출력 토큰 (SenseVoice / Whisper 특수 토큰)
# ───────────────────────────────────────────────────────────────────────
_MODEL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")
_HF_SPECIAL_RE = re.compile(r"<(?:s|/s|pad|unk|mask|sep|cls)>", re.IGNORECASE)

# ───────────────────────────────────────────────────────────────────────
# AI-Hub / 한국 ASR 이중 전사
#   "(7시)/(일곱시)"   →  앞=철자 전사 / 뒤=발음 전사
#   "(ARS)/(에이 알 에스)"
#   "(16 %)/(십 육 프로)"
# 정책: 본 프로젝트는 *철자 전사* (Whisper/SenseVoice 사전학습과 일치).
# ───────────────────────────────────────────────────────────────────────
_DOUBLE_TRANS_RE = re.compile(r"\(([^)/]+)\)\s*/\s*\(([^)]+)\)")

# ───────────────────────────────────────────────────────────────────────
# 잡음/이벤트 태그 (데이터셋마다 표기 다양)
#   AIHub:  (NO:), (SN:), (NR:), (BR:)
#   리턴제로/일부:  /(noise), /(laughter)
#   기타:   o/, l/, b/, @웃음, @잡음
# ───────────────────────────────────────────────────────────────────────
_NOISE_TAG_RE = re.compile(
    r"\([A-Z]{2,4}\s*:\s*\)"           # (NO:), (SN:)
    r"|/\([a-z]+\)"                    # /(noise)
    r"|[olbn]/"                        # o/  l/  b/  n/
    r"|@(?:잡음|웃음|기침|숨소리|박수)" # @웃음
)

# ───────────────────────────────────────────────────────────────────────
# 일반 구두점 (한국어 문장 기호 + ASCII)
# ───────────────────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(
    r"[.,!?;:\"'`~^*+_=<>\[\]{}()|/\\@#$%&　"
    r"“”‘’"        # 곡선 따옴표 “ ” ‘ ’
    r"「」『』"        # 「」『』
    r"～｟｠"              # 전각 ~ 등
    r"…‥·"              # …, ‥, ·
    r"、。，．"        # 、 。 ， ．
    r"※‼⁇⁈⁉"  # ※ ‼ ⁇ ⁈ ⁉
    r"]"
)

# 비표시 / 제어 문자 (zero-width, BOM 등)
_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁠-⁯﻿]")


def _strip_model_tokens(text: str) -> str:
    """모델 출력의 특수 토큰 (`<|ko|>`, `<|HAPPY|>`, `<s>` 등) 제거."""
    text = _MODEL_TOKEN_RE.sub("", text)
    text = _HF_SPECIAL_RE.sub("", text)
    return text


def _resolve_double_transcription(text: str, mode: Literal["spell", "phonetic"]) -> str:
    """이중 전사 `(A)/(B)` → A 또는 B 선택.

    Args:
        text: 입력.
        mode: 'spell' (앞=철자) or 'phonetic' (뒤=발음).
    """
    if mode == "spell":
        return _DOUBLE_TRANS_RE.sub(r"\1", text)
    elif mode == "phonetic":
        return _DOUBLE_TRANS_RE.sub(r"\2", text)
    raise ValueError(f"unknown mode: {mode!r}")


def _strip_noise_tags(text: str) -> str:
    """잡음/이벤트 태그 제거."""
    return _NOISE_TAG_RE.sub(" ", text)


def _strip_punct(text: str) -> str:
    """구두점 제거 (공백으로 치환)."""
    return _PUNCT_RE.sub(" ", text)


def _strip_invisible(text: str) -> str:
    """제어/비표시 문자 제거."""
    return _INVISIBLE_RE.sub("", text)


def _collapse_whitespace(text: str) -> str:
    """양끝 trim + 연속 공백 → 단일."""
    return " ".join(text.split())


def normalize_korean_asr(
    text: str,
    *,
    transcription: Literal["spell", "phonetic"] = "spell",
    lowercase: bool = True,
    strip_punct: bool = True,
    strip_noise: bool = True,
    strip_model_tokens: bool = True,
    nfc: bool = True,
) -> str:
    """한국어 ASR 전사 정규화 (표준 규칙).

    Args:
        text: 입력 텍스트 (raw transcript or 모델 출력).
        transcription: 이중 전사 `(A)/(B)` 처리 — `'spell'` 또는 `'phonetic'`.
        lowercase: 영문 소문자화.
        strip_punct: 구두점 제거.
        strip_noise: 잡음/이벤트 태그 제거.
        strip_model_tokens: 모델 특수 토큰 제거.
        nfc: 한글 NFC 정규화.

    Returns:
        정규화된 텍스트. 공백 정리 + 모든 옵션 적용.

    Examples:
        >>> normalize_korean_asr("안녕하세요.")
        '안녕하세요'
        >>> normalize_korean_asr("(7시)/(일곱시)에 만나요")
        '7시에 만나요'
        >>> normalize_korean_asr("(7시)/(일곱시)에 만나요", transcription="phonetic")
        '일곱시에 만나요'
        >>> normalize_korean_asr("Coffee 한 잔 (NO:) 주세요")
        'coffee 한 잔 주세요'
        >>> normalize_korean_asr("<|ko|><|HAPPY|>안녕하세요")
        '안녕하세요'
    """

    if not isinstance(text, str):
        text = str(text)

    if nfc:
        text = unicodedata.normalize("NFC", text)

    text = _strip_invisible(text)

    if strip_model_tokens:
        text = _strip_model_tokens(text)

    # 이중 전사는 *구두점 제거 전* 처리 (괄호 의존)
    text = _resolve_double_transcription(text, mode=transcription)

    if strip_noise:
        text = _strip_noise_tags(text)

    if strip_punct:
        text = _strip_punct(text)

    if lowercase:
        # 한글은 영향 없음. 영문/숫자만.
        text = text.lower()

    text = _collapse_whitespace(text)
    return text


# ───────────────────────────────────────────────────────────────────────
# 자모 분해 (보조 분석용)
# ───────────────────────────────────────────────────────────────────────
def to_jamo(text: str) -> str:
    """한글 음절을 자모 단위로 분해. CER 보조 분석 (받침/모음 오류 패턴).

    >>> to_jamo("강")
    'ㄱㅏㅇ'
    """
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:    # 한글 음절
            syllable_index = code - 0xAC00
            cho = syllable_index // 588
            jung = (syllable_index % 588) // 28
            jong = syllable_index % 28
            chos = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
            jungs = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
            jongs = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
            out.append(chos[cho])
            out.append(jungs[jung])
            if jong:
                out.append(jongs[jong])
        else:
            out.append(ch)
    return "".join(out)
