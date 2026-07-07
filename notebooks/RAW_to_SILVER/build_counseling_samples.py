#!/usr/bin/env python3
"""AIHub CounselingSpeech (KtelSpeech) Validation → transcript.jsonl 벤치마크 변환.

출력 구조:
  OUT_DIR/
    transcript.jsonl
    audio/<session>_<utt>.wav   (8kHz → 16kHz 업샘플, mono)

핵심 (앞 데이터셋과 다른 점):
  - 오디오 8kHz 전화망 → **16kHz로 업샘플**(-ar 16000). 음질 향상이 아니라 포맷 통일.
    (원본이 8kHz였다는 사실은 별도 기록 권장 — 필요시 source_sr 필드 추가)
  - 전사는 발화별 .txt(UTF-8), 세션 JSON(dialogs)으로 화자·메타 매핑.
  - 정규화는 KsponSpeech와 **동일 의미**:
      text            = clean_text(raw): 이중전사→발음형, 태그/마커 제거, 구두점 유지
      text_norm = text에서 구두점 제거 + 공백 정리
    + CounselingSpeech 추가: @(이름 마커) 기호 제거(단어 유지), (A)(B) 붙은 이중전사 처리.
  - 화자 메타(gender/age) 채움. ⚠ 전사에 PII(이름 등) 가능 — 처리·반출 주의.

DRY_RUN=True로 정규화 잔존(괄호/@/영문·숫자) 먼저 확인 → False로 변환.
"""
from pathlib import Path
import json, re, subprocess, random

try:
    import soundfile as sf
except ImportError:
    sf = None

# ========================= 설정 =========================
LABEL_ROOT = Path("/data/ASR/RAW/AIHub_CounselingSpeech/012.상담_음성_데이터"
                  "/01.데이터/2.Validation/라벨링데이터_1129_add")
OUT_DIR    = Path("/data/ASR/BENCHMARK/SILVER/AIHub_CounselingSpeech/valid")
CORPUS_ID  = "AIHub_CounselingSpeech"   # 데이터셋 식별자(평평). split은 폴더/키로 구분
SPLIT      = "valid"                     # 2.Validation 기준
TARGET_SR  = 16000          # 8kHz → 16kHz 업샘플
DUAL_MODE  = "pron"         # 이중전사: 'pron'=발음형 / 'spell'=표기형
MAX_SESSIONS = None          # 1차 확인용. 전체 변환 땐 None
DRY_RUN    = False

GENDER_MAP = {"여": "F", "남": "M"}

# ===================== 경로 매핑 =====================
def label_to_audio(p):
    return Path(str(p).replace("라벨링데이터", "원천데이터").replace("_label_", "_wav_"))

def read_txt(p):
    return Path(p).read_text(encoding="utf-8", errors="replace").strip()

# ======================= 정규화 =======================
RE_DUAL  = re.compile(r"\(([^)]*)\)/\(([^)]*)\)")     # (표기)/(발음)
RE_DUAL2 = re.compile(r"\(([^()]*)\)\(([^()]*)\)")    # (표기)(발음)  예: (10)(십)
RE_NOISE = re.compile(r"(?:^|\s)[bnlou]/")            # 비언어 태그 b/ n/ l/ o/ u/
RE_PUNCT = re.compile(r"[.,!?]")                      # text_norm에서 제거(KsponSpeech와 동일)

def clean_text(raw, dual_mode=DUAL_MODE):
    """text: 이중전사 해소·태그/마커 제거, 구두점 유지."""
    t = raw.strip()
    pick = (lambda m: m.group(1)) if dual_mode == "spell" else (lambda m: m.group(2))
    t = RE_DUAL.sub(pick, t)       # (A)/(B) 먼저
    t = RE_DUAL2.sub(pick, t)      # (A)(B) 그다음
    t = RE_NOISE.sub(" ", t)       # 비언어 태그 삭제
    t = t.replace("@", "")         # PII 이름 마커 기호만 제거(단어는 발화되므로 유지)
    t = t.replace("/", "")         # 간투어 슬래시('어/' '막/')의 '/'만 제거
    t = t.replace("+", " ").replace("*", " ")   # 자가수정/오발음 마커 제거
    return re.sub(r"\s+", " ", t).strip()

def normalize_text(text):
    """text_norm: 구두점 제거 + 공백 정리."""
    return re.sub(r"\s+", " ", RE_PUNCT.sub("", text)).strip()

# 필수 키 외 옵션 필드는 값이 없으면(None) 키 자체를 제거
REQUIRED_KEYS = {"key", "audio", "duration", "text", "text_norm"}
def prune(rec):
    return {k: v for k, v in rec.items() if k in REQUIRED_KEYS or v is not None}

# ======================= 오디오 =======================
def wav_duration(p):
    i = sf.info(str(p)); return round(i.frames / i.samplerate, 3)

def to_wav16(src, dst):
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", str(TARGET_SR), "-ac", "1", str(dst)],
                   check=True, capture_output=True)

# ===================== manifest =====================
def build_rows(label_root, max_sessions):
    rows = []
    jsons = sorted(label_root.rglob("S*.json"))
    if max_sessions:
        jsons = jsons[:max_sessions]
    for jp in jsons:
        try:
            ds = json.loads(jp.read_text(encoding="utf-8", errors="replace"))["dataSet"]
        except Exception:
            continue
        ti  = ds.get("typeInfo", {})
        spk = {s["id"]: s for s in ti.get("speakers", [])}
        domain  = ti.get("category")
        session = jp.stem
        audio_dir = label_to_audio(jp.parent)
        for d in ds.get("dialogs", []):
            sid = d.get("speaker"); sm = spk.get(sid, {})
            txt_name = Path(d["textPath"]).name
            wav_name = Path(d["audioPath"]).name
            tf = jp.parent / txt_name
            if not tf.exists():
                continue
            raw = read_txt(tf)
            rows.append({
                "session": session, "utt": Path(txt_name).stem, "domain": domain,
                "speaker": sid, "spk_type": sm.get("type"),
                "gender": sm.get("gender"), "age": sm.get("age"),
                "raw": raw, "src_wav": audio_dir / wav_name,
            })
    return rows

# ========================= 메인 =========================
def main():
    if sf is None:
        print("⚠ soundfile 필요: pip install soundfile"); return
    rows = build_rows(LABEL_ROOT, MAX_SESSIONS)
    print(f"세션 {len({r['session'] for r in rows}):,} / 발화 {len(rows):,}  "
          f"(MAX_SESSIONS={MAX_SESSIONS}, TARGET_SR={TARGET_SR})")

    fout = None
    if not DRY_RUN:
        (OUT_DIR / "audio").mkdir(parents=True, exist_ok=True)
        fout = open(OUT_DIR / "transcript.jsonl", "w", encoding="utf-8")

    records, miss = [], 0
    for i, r in enumerate(rows):
        raw  = r["raw"]
        name = f"{r['session']}_{r['utt']}"
        rec = {
            "key": f"{CORPUS_ID}_{SPLIT}__{i:05d}",
            "audio": f"audio/{name}.wav",
            "duration": None,
            "text": raw,                                       # 원본 전사 그대로
            "text_norm": clean_text(raw),   # 정리형(구두점 유지, 신규 포맷)
            "age": r["age"],
            "gender": GENDER_MAP.get(r["gender"]),
            # CounselingSpeech 부가 메타(채점기는 미사용 키 무시)
            "spk_type": r["spk_type"],
            "domain": r["domain"],
        }
        rec = prune(rec)
        if not DRY_RUN:
            src = r["src_wav"]
            if not src.exists():
                miss += 1; continue
            dst = OUT_DIR / "audio" / f"{name}.wav"
            if not dst.exists():
                to_wav16(src, dst)        # 8k → 16k 업샘플
            rec["duration"] = wav_duration(dst)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.append(rec)

    if not DRY_RUN:
        fout.close()
        print(f"완료: {len(records)-miss:,}건 (누락 {miss}) → {OUT_DIR/'transcript.jsonl'}")

    # 정규화 잔존 점검: text_norm에 정리 안 된 게 남았는지 (text는 원본이라 당연히 마커 있음)
    paren = [r for r in records if re.search(r"[()]", r["text_norm"])]
    at    = [r for r in records if "@" in r["text_norm"]]
    bare  = [r for r in records if re.search(r"[A-Za-z0-9]", r["text_norm"])]
    print(f"\n[정규화 잔존: text_norm] 괄호 {len(paren)} / @ {len(at)} / 영문·숫자 {len(bare)} (총 {len(records)})")
    for tag, lst in [("괄호", paren), ("영문·숫자", bare)]:
        for r in lst[:3]:
            print(f"   ({tag}) {r['text_norm'][:70]}")

    if DRY_RUN:
        print("\n--- 레코드 미리보기 2건 ---")
        for r in records[:2]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        print("\nDRY_RUN=True → 변환/쓰기 생략. 잔존 확인 후 False로 실행.")

if __name__ == "__main__":
    main()