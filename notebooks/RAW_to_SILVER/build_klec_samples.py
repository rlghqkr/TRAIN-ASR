#!/usr/bin/env python3
"""AIHub KorLectureSpeech (KlecSpeech) Validation → transcript.jsonl 변환.

출력 구조:
  OUT_DIR/
    transcript.jsonl
    audio/<session>_<utt>.wav

EDA 확인 사항 반영:
  - 구조 = CounselingSpeech(①형): 세션 JSON dialogs + 발화별 txt/wav, _label_→_wav_ 매핑
  - **오디오 이미 16kHz mono PCM_16 → 복사만** (ffmpeg 불필요)
  - 전사 KsponSpeech 풀세트: (A)/(B) 12,445, n/ b/ o/ u/ l/, +, *, @, (A)(B)
    → 기존 정규화 재사용 (이중전사→발음형, 태그/마커 제거, 구두점 유지)
  - duration: JSON에 없음 → sf.info(헤더만)로 측정
  - age_raw 지저분('None' 문자열, '30(?)', '30', '60대') → 숫자 추출해 "30대" 통일, 불명은 prune
  - valid 94 JSON 중 일부 파싱 실패 가능 → 카운트 출력

DRY_RUN=True로 잔존·age 정리 확인 → False로 전체 변환.
"""
from pathlib import Path
import json, re, shutil

try:
    import soundfile as sf
except ImportError:
    sf = None

# ========================= 설정 =========================
LABEL_ROOT = Path("/data/ASR/RAW/AIHub_KorLectureSpeech/009.한국어_강의_데이터"
                  "/01.데이터/2.Validation/라벨링데이터_0908_add")
OUT_DIR    = Path("/data/ASR/BENCHMARK/SILVER/AIHub_KorLectureSpeech/valid")
CORPUS_ID  = "AIHub_KorLectureSpeech"
SPLIT      = "valid"
DUAL_MODE  = "pron"          # 이중전사: 발음형
MAX_SESSIONS = None            # 1차 확인용. 전체는 None
DRY_RUN    = False

GENDER_MAP = {"여": "F", "남": "M"}

# ===================== 경로 매핑 =====================
def label_to_audio(p):
    return Path(str(p).replace("라벨링데이터", "원천데이터").replace("_label_", "_wav_"))

def read_txt(p):
    return Path(p).read_text(encoding="utf-8", errors="replace").strip()

# ======================= 정규화 (KsponSpeech 재사용) =======================
RE_DUAL  = re.compile(r"\(([^)]*)\)/\(([^)]*)\)")     # (표기)/(발음)
RE_DUAL2 = re.compile(r"\(([^()]*)\)\(([^()]*)\)")    # (표기)(발음)
RE_NOISE = re.compile(r"(?:^|\s)[bnlou]/")            # 비언어 태그

def clean_text(raw, dual_mode=DUAL_MODE):
    """text_norm: 이중전사 해소·태그/마커 제거. 구두점(?!.,) 유지."""
    t = raw.strip()
    pick = (lambda m: m.group(1)) if dual_mode == "spell" else (lambda m: m.group(2))
    t = RE_DUAL.sub(pick, t)
    t = RE_DUAL2.sub(pick, t)
    t = RE_NOISE.sub(" ", t)
    t = t.replace("@", "")                       # 이름 마커 기호 제거(단어 유지)
    t = t.replace("/", "")                       # 간투어 슬래시('자/' '이/') 제거(단어 유지)
    t = t.replace("+", " ").replace("*", " ")    # 자가수정/오발음 마커 제거
    return re.sub(r"\s+", " ", t).strip()

# ----- age 정리: '30(?)'/'30'/'60대' → '30대'/'60대', 'None'/불명 → None -----
RE_AGE = re.compile(r"(\d+)")
def clean_age(raw):
    if not raw or str(raw).strip().lower() == "none":
        return None
    m = RE_AGE.search(str(raw))
    if not m:
        return None
    n = int(m.group(1))
    return f"{n}대" if n >= 10 else None

# 필수 키 외 옵션 필드는 값이 없으면(None) 키 자체를 제거
REQUIRED_KEYS = {"key", "audio", "duration", "text", "text_norm"}
def prune(rec):
    return {k: v for k, v in rec.items() if k in REQUIRED_KEYS or v is not None}

# ===================== manifest =====================
def build_rows(label_root, max_sessions):
    rows, bad = [], 0
    jsons = sorted(label_root.rglob("S*.json"))
    if max_sessions:
        jsons = jsons[:max_sessions]
    for jp in jsons:
        try:
            ds = json.loads(jp.read_text(encoding="utf-8", errors="replace"))["dataSet"]
        except Exception:
            bad += 1; continue
        ti  = ds.get("typeInfo", {})
        spk = {s["id"]: s for s in ti.get("speakers", [])}
        audio_dir = label_to_audio(jp.parent)
        dom, grp = jp.parents[2].name, jp.parents[1].name   # D##, G##/M## (세션 ID가 도메인 간 중복되므로 필수)
        for d in ds.get("dialogs", []):
            sm = spk.get(d.get("speaker"), {})
            txt_name = Path(d["textPath"]).name
            wav_name = Path(d["audioPath"]).name
            tf = jp.parent / txt_name
            if not tf.exists():
                continue
            rows.append({
                "session": jp.stem, "dom": dom, "grp": grp,
                "utt": Path(txt_name).stem,
                "gender": sm.get("gender"), "age_raw": sm.get("age"),
                "category": ti.get("category"), "subcat": ti.get("subcategory"),
                "raw": read_txt(tf),
                "src_wav": audio_dir / wav_name,
            })
    return rows, bad, len(jsons)

# ========================= 메인 =========================
def main():
    if sf is None:
        print("⚠ soundfile 필요"); return
    rows, bad, n_json = build_rows(LABEL_ROOT, MAX_SESSIONS)
    n_sess = len({(r["dom"], r["grp"], r["session"]) for r in rows})
    print(f"JSON {n_json:,}개 중 파싱실패 {bad} → 세션 {n_sess:,} / 발화 {len(rows):,}  "
          f"(MAX_SESSIONS={MAX_SESSIONS}, DRY_RUN={DRY_RUN})")

    fout = None
    if not DRY_RUN:
        (OUT_DIR / "audio").mkdir(parents=True, exist_ok=True)
        fout = open(OUT_DIR / "transcript.jsonl", "w", encoding="utf-8")

    records, miss = [], 0
    for i, r in enumerate(rows):
        name = f"{r['dom']}_{r['grp']}_{r['session']}_{r['utt']}"   # 세션ID가 D## 간 중복되므로 경로 포함
        rec = {
            "key": f"{CORPUS_ID}_{SPLIT}__{i:05d}",
            "audio": f"audio/{name}.wav",
            "duration": None,
            "text": r["raw"],                      # 원본 전사 그대로
            "text_norm": clean_text(r["raw"]),     # KsponSpeech 규칙(구두점 유지)
            "age": clean_age(r["age_raw"]),
            "gender": GENDER_MAP.get(r["gender"]),
            # ----- 추가 메타 -----
            "domain": r["category"],
            "subcat": r["subcat"],
        }
        rec = prune(rec)
        records.append(rec)
        if not DRY_RUN:
            src = r["src_wav"]
            if not src.exists():
                miss += 1; continue
            dst = OUT_DIR / "audio" / f"{name}.wav"
            if not dst.exists():
                shutil.copy2(src, dst)             # 16k 그대로 복사 (재개 가능)
            info = sf.info(str(dst))
            rec["duration"] = round(info.frames / info.samplerate, 3)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not DRY_RUN:
        fout.close()
        print(f"완료: {len(records)-miss:,}건 (오디오 누락 {miss}) → {OUT_DIR/'transcript.jsonl'}")

    # 잔존 점검 (text_norm; 구두점은 유지가 정상)
    paren = [r for r in records if re.search(r"[()]", r["text_norm"])]
    bare  = [r for r in records if re.search(r"[A-Za-z0-9]", r["text_norm"])]
    age_none = sum(1 for r in records if "age" not in r)
    print(f"\n[잔존: text_norm] 괄호 {len(paren)} / 영문·숫자 {len(bare)} | age 불명(키 제거) {age_none} (총 {len(records):,})")
    for tag, lst in [("괄호", paren)]:
        for r in lst[:3]:
            print(f"   ({tag}) {r['text_norm'][:70]}")

    if DRY_RUN:
        print("\n--- 레코드 미리보기 2건 ---")
        for r in records[:2]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        print("\nDRY_RUN=True → 복사/쓰기 생략. 확인 후 MAX_SESSIONS=None, DRY_RUN=False로 실행.")

if __name__ == "__main__":
    main()