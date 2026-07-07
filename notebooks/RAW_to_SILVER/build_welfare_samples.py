#!/usr/bin/env python3
"""AIHub WelfareCounsel (복지분야 콜상담) Validation → transcript.jsonl 변환.

출력 구조:
  OUT_DIR/
    transcript.jsonl
    audio/<utt_id>.wav

이 데이터셋 특징 (EDA 확인):
  - 발화 단위 JSON (orgtext + metadata). wav는 경로 치환(라벨링→원천, VL→VS)으로 매핑.
  - **오디오 이미 16kHz mono PCM_16 → 리샘플 없이 복사만** (ffmpeg 불필요, 빠름)
  - duration = metadata.sptime_all (파일 길이와 차이 0.000s 확인)
  - 전사 깨끗(태그·이중전사 없음, 구두점 정문) → text_norm은 공백 정리 수준
    · 'ㅇㅇ' = 이름 익명화 마스킹 → 그대로 유지 (음성-전사 불일치 인지)
  - valid 223,546발화. SAMPLE_N으로 균형 샘플 가능(None=전수)
  - ⚠ train/valid 화자 겹침 큼 (speaker-disjoint 아님 — 평가 해석 시 유의)

DRY_RUN=True + MAX_FILES로 확인 → 전수/샘플 결정 후 실행.
"""
from pathlib import Path
import json, re, random, shutil

# ========================= 설정 =========================
LABEL_ROOT = Path("/data/ASR/RAW/AIHub_WelfareCounsel/01.데이터/2.Validation/라벨링데이터")
OUT_DIR    = Path("/data/ASR/BENCHMARK/SILVER/AIHub_WelfareCounsel/valid")
CORPUS_ID  = "AIHub_WelfareCounsel"
SPLIT      = "valid"
MAX_FILES  = 2000      # DRY 확인용 표본. 전수는 None
SAMPLE_N   = None      # 전수 변환이면 None, 샘플 벤치마크면 개수(랜덤 시드 고정)
DRY_RUN    = True

GENDER_MAP = {"여": "F", "남": "M"}

# ===================== 경로 매핑 =====================
def json_to_wav(jp):
    s = str(jp).replace("라벨링데이터", "원천데이터").replace("/VL", "/VS").replace("/TL", "/TS")
    return Path(s).with_suffix(".wav")

# ======================= 정규화 =======================
def clean_text(raw):
    """text_norm: 이 데이터셋은 전사가 깨끗 → 공백 정리만. 구두점 유지. ㅇㅇ(익명화)도 유지."""
    return re.sub(r"\s+", " ", raw.strip())

# 필수 키 외 옵션 필드는 값이 없으면(None) 키 자체를 제거
REQUIRED_KEYS = {"key", "audio", "duration", "text", "text_norm"}
def prune(rec):
    return {k: v for k, v in rec.items() if k in REQUIRED_KEYS or v is not None}

# ===================== manifest =====================
def build_rows(label_root, max_files, sample_n):
    rows, bad = [], 0
    jsons = sorted(label_root.rglob("*.json"))
    total = len(jsons)
    if max_files and total > max_files:
        jsons = random.Random(0).sample(jsons, max_files)
    elif sample_n and total > sample_n:
        jsons = random.Random(0).sample(jsons, sample_n)
    for jp in jsons:
        try:
            d  = json.loads(jp.read_text(encoding="utf-8", errors="replace"))
            md = d["info"][0]["metadata"]
            text = d["inputText"][0]["orgtext"]
        except Exception:
            bad += 1; continue
        rows.append({
            "utt": jp.stem,
            "text": (text or "").strip(),
            "duration": float(md["sptime_all"]) if md.get("sptime_all") else None,
            "gender": md.get("speaker_sex"), "age": md.get("speaker_age"),
            "spk_type": md.get("speaker_type"), "domain": md.get("category1"),
            "src_wav": json_to_wav(jp),
        })
    return rows, bad, total

# ========================= 메인 =========================
def main():
    rows, bad, total = build_rows(LABEL_ROOT, MAX_FILES if DRY_RUN else None, SAMPLE_N)
    print(f"전체 JSON {total:,} 중 처리 {len(rows):,} (파싱실패 {bad})  "
          f"[MAX_FILES={MAX_FILES if DRY_RUN else None}, SAMPLE_N={SAMPLE_N}, DRY_RUN={DRY_RUN}]")

    fout = None
    if not DRY_RUN:
        (OUT_DIR / "audio").mkdir(parents=True, exist_ok=True)
        fout = open(OUT_DIR / "transcript.jsonl", "w", encoding="utf-8")

    records, miss = [], 0
    for i, r in enumerate(rows):
        rec = {
            "key": f"{CORPUS_ID}_{SPLIT}__{i:06d}",
            "audio": f"audio/{r['utt']}.wav",
            "duration": r["duration"],
            "text": r["text"],                    # 원본 전사 그대로 (ㅇㅇ 마스킹 포함)
            "text_norm": clean_text(r["text"]),   # 공백 정리(구두점 유지)
            "age": r["age"],
            "gender": GENDER_MAP.get(r["gender"]),
            # ----- 추가 메타 -----
            "spk_type": r["spk_type"],
            "domain": r["domain"],
        }
        rec = prune(rec)
        records.append(rec)
        if not DRY_RUN:
            src = r["src_wav"]
            if not src.exists():
                miss += 1; continue
            dst = OUT_DIR / "audio" / f"{r['utt']}.wav"
            if not dst.exists():
                shutil.copy2(src, dst)     # 이미 16k → 복사만 (재개 가능)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not DRY_RUN:
        fout.close()
        print(f"완료: {len(records)-miss:,}건 (오디오 누락 {miss}) → {OUT_DIR/'transcript.jsonl'}")

    # 점검: 빈 전사/duration 결측
    empty = sum(1 for r in records if not r["text"])
    nodur = sum(1 for r in records if r.get("duration") is None)
    print(f"\n[점검] 빈 전사 {empty} / duration 결측 {nodur} (총 {len(records):,})")

    if DRY_RUN:
        print("\n--- 레코드 미리보기 2건 ---")
        for r in records[:2]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        print("\nDRY_RUN=True → 복사/쓰기 생략. SAMPLE_N(전수 vs 샘플) 결정 후 DRY_RUN=False로 실행.")

if __name__ == "__main__":
    main()