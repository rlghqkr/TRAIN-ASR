#!/usr/bin/env python3
"""KsponSpeech eval(eval_clean, eval_other)를 split별로 transcript.jsonl 벤치마크 포맷 변환.

출력 구조 (split을 각각 분리):
  OUT_DIR/
    eval_clean/
      transcript.jsonl
      audio/KsponSpeech_Exxxxx.wav
    eval_other/
      transcript.jsonl
      audio/KsponSpeech_Exxxxx.wav

PCM(16k/16bit/mono raw) → WAV(동일 규격, 컨테이너만 변경)이라 리샘플링 없음.
처음엔 DRY_RUN=True로 미리보기만 하고, 정규화 규칙 확정 후 False로 실행할 것.
"""
from pathlib import Path
import json, re, subprocess

# ========================= 설정 =========================
ROOT    = Path("/data/ASR/RAW/AIHub_KsponSpeech/10.한국어음성")
SCRIPTS = ROOT / "KsponSpeech_scripts"
OUT_DIR = Path("/data/ASR/BENCHMARK/SILVER/AIHub_KsponSpeech/KsponSpeech_eval")  # SILVER/AIHub_KsponSpeech 아래
SPLITS  = {
    "eval_clean": SCRIPTS / "eval_clean.trn",
    "eval_other": SCRIPTS / "eval_other.trn",
}
SR, CH, SAMPW = 16000, 1, 2
DUAL_MODE = "pron"    # reading form 기준: 이중전사 (표기)/(발음) → '발음형' 채택. 'spell'은 표기형
DRY_RUN   = False      # True면 변환·쓰기 없이 split별 미리보기만

# ===================== 인코딩 안전 읽기 =====================
def read_text_file(p):
    b = Path(p).read_bytes()
    for enc in ("cp949", "utf-8", "euc-kr"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", "replace")

# ======================= 정규화 =======================
RE_DUAL  = re.compile(r"\(([^)]*)\)/\(([^)]*)\)")   # (표기)/(발음)
RE_NOISE = re.compile(r"(?:^|\s)[bnlou]/")          # 비언어 태그 b/ n/ l/ o/ u/
RE_PUNCT = re.compile(r"[.,!?]")

def clean_text(raw, dual_mode=DUAL_MODE):
    """text_norm: 태그·마커 제거, 이중전사 해소. 구두점(?!.,)은 유지(신규 포맷)."""
    t = raw.strip()
    t = RE_DUAL.sub(lambda m: m.group(1) if dual_mode == "spell" else m.group(2), t)
    t = RE_NOISE.sub(" ", t)                    # 비언어 태그 삭제
    t = t.replace("/", "")                      # 간투어 슬래시(어/ 그/ 막/)의 '/'만 제거(단어 유지)
    t = t.replace("+", " ").replace("*", " ")   # 자가수정/오발음 마커 제거(단어 유지)
    return re.sub(r"\s+", " ", t).strip()

def normalize_text(text):
    """비교용 'text_norm': 구두점 제거 + 공백 정리."""
    return re.sub(r"\s+", " ", RE_PUNCT.sub("", text)).strip()

# 필수 키 외 옵션 필드는 값이 없으면(None) 키 자체를 제거
REQUIRED_KEYS = {"key", "audio", "duration", "text", "text_norm"}
def prune(rec):
    return {k: v for k, v in rec.items() if k in REQUIRED_KEYS or v is not None}

# ======================= 오디오 =======================
def pcm_duration(p):
    return round((Path(p).stat().st_size // SAMPW) / SR, 3)

def pcm_to_wav(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", str(CH),
         "-i", str(src), str(dst)],
        check=True, capture_output=True,
    )

# ===================== split 단위 처리 =====================
def process_split(split, trn, write):
    """한 split을 OUT_DIR/<split>/ 로 변환. write=False면 미리보기용 레코드만 생성."""
    out_dir   = OUT_DIR / split
    corpus_id = "AIHub_KsponSpeech"      # 데이터셋 식별자(평평). split은 폴더/키로 구분
    lines = [l for l in read_text_file(trn).splitlines() if "::" in l]

    fout = None
    if write:
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
        fout = open(out_dir / "transcript.jsonl", "w", encoding="utf-8")

    records, miss = [], 0
    for i, line in enumerate(lines):
        rel, raw = (s.strip() for s in line.split("::", 1))
        src_pcm = ROOT / rel
        stem = Path(rel).stem
        rec = {
            "key": f"{corpus_id}_{split}__{i:05d}",
            "audio": f"audio/{stem}.wav",      # 이 split 폴더 기준 상대경로
            "duration": pcm_duration(src_pcm) if src_pcm.exists() else None,
            "text": raw,                                       # 원본 전사 그대로
            "text_norm": clean_text(raw),   # 정리형(구두점 유지, 신규 포맷)
            "age": None,           # 메타 없음 → prune에서 키 제거됨
            "gender": None,        # 메타 없음 → prune에서 키 제거됨
        }
        rec = prune(rec)
        records.append(rec)
        if write:
            if not src_pcm.exists():
                miss += 1
                continue
            dst = out_dir / "audio" / f"{stem}.wav"
            if not dst.exists():
                pcm_to_wav(src_pcm, dst)       # 재개 가능
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if write:
        fout.close()
        print(f"[{split}] {len(records):,}건, 누락 {miss}건 → {out_dir / 'transcript.jsonl'}")
    return records

# ========================= 메인 =========================
def main():
    for split, trn in SPLITS.items():
        if not trn.exists():
            print(f"[경고] 없음: {trn}")
            continue
        print(f"\n===== {split} =====")
        records = process_split(split, trn, write=not DRY_RUN)
        print(f"총 {len(records):,} 발화")
        # reading form 점검: 이중전사가 아닌 bare 숫자/영문은 자동 변환 안 됨 → 잔존량 확인
        resid = [r for r in records if re.search(r"[0-9A-Za-z]", r["text_norm"])]
        print(f"  ※ text_norm에 숫자/영문 잔존(bare, 수동 처리 필요): "
              f"{len(resid):,}건 ({len(resid)/max(len(records),1)*100:.2f}%)")
        for r in resid[:3]:
            print(f"     예) {r['text_norm'][:60]}")
        if DRY_RUN:
            print("--- 미리보기 2건 ---")
            for rec in records[:2]:
                print(json.dumps(rec, ensure_ascii=False, indent=2))

    if DRY_RUN:
        print("\nDRY_RUN=True → 변환/쓰기 생략. 정규화 규칙 확정 후 False로 바꿔 실행.")

if __name__ == "__main__":
    main()