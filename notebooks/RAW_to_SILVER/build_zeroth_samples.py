#!/usr/bin/env python3
"""OpenSLR SLR40 Zeroth-Korean → transcript.jsonl 벤치마크 포맷 변환.

출력 구조 (split별 분리):
  OUT_DIR/
    test/
      transcript.jsonl
      audio/<utt_id>.wav

Zeroth 특성:
  - 오디오 FLAC(16k/mono/16bit) → WAV로 transcode (-ar 16000 -ac 1)
  - 전사가 깨끗(구두점·숫자·영문·태그 없음) → 정규화 거의 no-op
  - AUDIO_INFO(SPEAKERID|NAME|SEX|SCRIPTID|DATASET)로 gender 채움
  - test는 train과 speaker-disjoint → 평가셋으로 적합

처음엔 DRY_RUN=True로 미리보기만, 확인 후 False로 실행.
"""
from pathlib import Path
import json, re, subprocess

try:
    import soundfile as sf
except ImportError:
    sf = None

# ========================= 설정 =========================
ROOT       = Path("/data/ASR/RAW/OpenSLR_SLR40_Zeroth_ko")
AUDIO_INFO = ROOT / "AUDIO_INFO"
OUT_DIR    = Path("/data/ASR/BENCHMARK/SILVER/OpenSLR_SLR40_Zeroth_ko")
SPLITS = {
    "test":  ROOT / "test_data_01",
    # "train": ROOT / "train_data_01",   # 학습셋도 SILVER로 만들려면 주석 해제(22k개, 오래 걸림)
}
DRY_RUN = False

# ===================== 화자 메타 (AUDIO_INFO) =====================
def load_speaker_gender(audio_info):
    """SPEAKERID|NAME|SEX|SCRIPTID|DATASET → {speaker_id: 'male'/'female'}"""
    m = {}
    if not Path(audio_info).exists():
        print(f"[경고] AUDIO_INFO 없음: {audio_info}")
        return m
    lines = Path(audio_info).read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[1:]:                       # 첫 줄은 헤더
        parts = line.split("|")
        if len(parts) < 3:
            continue
        spk, sex = parts[0].strip(), parts[2].strip().lower()
        m[spk] = {"m": "M", "f": "F"}.get(sex)
    return m

SPK_GENDER = load_speaker_gender(AUDIO_INFO)

# ======================= 정규화 =======================
RE_PUNCT = re.compile(r"[.,!?]")

def normalize_text(t):
    """Zeroth는 이미 깨끗 → 공백 정리만 (구두점은 신규 포맷상 유지, 원래 없음)."""
    return re.sub(r"\s+", " ", t).strip()

# 필수 키 외 옵션 필드는 값이 없으면(None) 키 자체를 제거
REQUIRED_KEYS = {"key", "audio", "duration", "text", "text_norm"}
def prune(rec):
    return {k: v for k, v in rec.items() if k in REQUIRED_KEYS or v is not None}

# ======================= 오디오 =======================
def flac_duration(p):
    i = sf.info(str(p))
    return round(i.frames / i.samplerate, 3)

def flac_to_wav(src, dst):
    # FLAC → WAV 16k mono (Zeroth가 이미 16k/mono라 transcode만)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True, capture_output=True,
    )

# ===================== manifest =====================
def build_rows(split_dir):
    """모든 .trans.txt 파싱 → [(utt_id, speaker, text, flac_path)]"""
    rows = []
    for trans in sorted(Path(split_dir).rglob("*.trans.txt")):
        d = trans.parent
        for line in trans.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            uid, text = parts[0], parts[1].strip()
            rows.append((uid, uid.split("_")[0], text, d / f"{uid}.flac"))
    return rows

# ===================== split 단위 처리 =====================
def process_split(split, split_dir, write):
    out_dir   = OUT_DIR / split
    corpus_id = "OpenSLR_SLR40_Zeroth_ko"   # 데이터셋 식별자(평평). split은 폴더/키로 구분
    rows = build_rows(split_dir)

    fout = None
    if write:
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
        fout = open(out_dir / "transcript.jsonl", "w", encoding="utf-8")

    records, miss, no_gender = [], 0, 0
    for i, (uid, spk, text, flac) in enumerate(rows):
        gender = SPK_GENDER.get(spk)
        if gender is None:
            no_gender += 1
        rec = {
            "key": f"{corpus_id}_{split}__{i:05d}",
            "audio": f"audio/{uid}.wav",
            "duration": flac_duration(flac) if flac.exists() else None,
            "text": text,
            "text_norm": normalize_text(text),
            "age": None,                  # Zeroth 미제공
            "gender": gender,             # AUDIO_INFO의 SEX → F/M
        }
        rec = prune(rec)
        records.append(rec)
        if write:
            if not flac.exists():
                miss += 1
                continue
            dst = out_dir / "audio" / f"{uid}.wav"
            if not dst.exists():
                flac_to_wav(flac, dst)    # 재개 가능
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if write:
        fout.close()
        print(f"[{split}] {len(records):,}건, 누락 {miss}건, 성별없음 {no_gender}건 "
              f"→ {out_dir / 'transcript.jsonl'}")
    return records, no_gender

# ========================= 메인 =========================
def main():
    if sf is None:
        print("⚠ soundfile 필요: pip install soundfile"); return
    for split, sd in SPLITS.items():
        if not sd.exists():
            print(f"[경고] 없음: {sd}")
            continue
        print(f"\n===== {split} =====")
        records, no_gender = process_split(split, sd, write=not DRY_RUN)
        print(f"총 {len(records):,} 발화 / 성별 매핑 안 된 화자 발화 {no_gender}건")
        if DRY_RUN:
            print("--- 미리보기 2건 ---")
            for rec in records[:2]:
                print(json.dumps(rec, ensure_ascii=False, indent=2))

    if DRY_RUN:
        print("\nDRY_RUN=True → 변환/쓰기 생략. 확인 후 False로 바꿔 실행.")

if __name__ == "__main__":
    main()