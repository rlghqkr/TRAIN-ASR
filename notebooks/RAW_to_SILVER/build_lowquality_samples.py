#!/usr/bin/env python3
"""AIHub LowQualityPhoneVoice (저음질 전화망) Validation → transcript.jsonl 변환 (수정본).

이전 버전 대비 바뀐 점 (디버깅으로 확정된 사항):
  [치명적 버그 수정] 세션 일련번호 S###### 는 카테고리(D**/J**) '안에서만' 유일하다.
    이전 코드는 jp.stem(=세션)만으로 출력명을 지어 D**/J** 를 버렸고, 그 결과 서로 다른
    도메인의 동명 세션(예: VL_D01/D01/J020/S023370 와 VL_D04/D04/J18/S023370)이
    같은 출력 파일명 'S023370_0001.wav' 로 충돌 → `if not dst.exists()` 때문에 뒤 발화가
    앞 발화 오디오를 재사용 → 텍스트≠오디오 오매핑(2,722건).
    → uid = "<D**>_<J**>_<S######>_<utt>" 로 카테고리를 포함해 충돌을 원천 제거.
      (서브카테고리 자릿수 불규칙: J18 / J020 → 정규식 J\\d+ 로 대응)

매핑 정책 (원본 유지 — 이게 표준):
  text       = 원본 전사 그대로 (태그·이중전사 포함)
  text_norm  = 정리형 (이중전사→발음형, 태그 제거). 구두점(?!.,)은 '유지' (기존 동작 그대로)
  duration   = JSON dialogs.duration 제공값 사용 (파일 길이와 차이 ≤0.005s 확인됨)

견고성 추가: ffmpeg 변환 per-file try/except, 원본누락(miss)/변환실패(fail) 분리 집계,
            uid 충돌 게이트, 사후 검증(audio/key 중복·도메인 분포).

DRY_RUN=True 로 uid 중복 0 / 잔존 확인 후 False 로 전체 변환.
출력:
  OUT_DIR/transcript.jsonl
  OUT_DIR/audio/<D**>_<J**>_<S######>_<utt>.wav   (8kHz → 16kHz 업샘플, mono)
"""
from pathlib import Path
from collections import Counter
import json
import re
import subprocess
import shutil

# ========================= 설정 =========================
LABEL_ROOT = Path("/data/ASR/RAW/AIHub_LowQualityPhoneVoice/007.저음질_전화망_음성인식_데이터"
                  "/01.데이터/2.Validation/라벨링데이터_230316")
OUT_DIR    = Path("/data/ASR/BENCHMARK/SILVER/AIHub_LowQualityPhoneVoice/valid")
CORPUS_ID  = "AIHub_LowQualityPhoneVoice"
SPLIT      = "valid"
TARGET_SR  = 16000
DUAL_MODE  = "pron"          # 이중전사: 'pron'=발음형 / 'spell'=표기형
MAX_SESSIONS = None           # 1차 확인용. 전체 변환 땐 None
DRY_RUN    = True            # 먼저 True 로 검증 → False 로 변환
CLEAN_AUDIO = False          # True 면 변환 전 기존 audio/ 삭제 (옛 스킴 잔재 제거 권장)

GENDER_MAP = {"여": "F", "남": "M"}

# 경로 계층 검증 (.../<Dxx>/<Jxx+>/<S######>/<S######>.json)
RE_DCAT = re.compile(r"^D\d+$")
RE_JCAT = re.compile(r"^J\d+$")
RE_SESS = re.compile(r"^S\d+$")

# ===================== 경로 매핑 =====================
def label_to_audio_base(tl_dir):
    """라벨 VL_D##(TL_D##) 디렉터리 → 원천 VS_D##(TS_D##)"""
    return Path(str(tl_dir).replace("라벨링데이터", "원천데이터")
                          .replace("TL_", "TS_").replace("VL_", "VS_"))

def parse_categories(jp):
    """JSON 파일의 디스크 경로에서 (D**, J**, S######) 추출. 형식 불일치 시 None."""
    try:
        session = jp.parent.name
        jcat    = jp.parents[1].name
        dcat    = jp.parents[2].name
    except Exception:
        return None
    if not (RE_DCAT.match(dcat) and RE_JCAT.match(jcat) and RE_SESS.match(session)):
        return None
    return dcat, jcat, session

# ======================= 정규화 =======================
# 처리 순서가 중요:
#   1) 이중전사 (A)/(B), (A)(B) → 발음형 선택   (괄호 구조부터 해소)
#   2) (( )) 마커 → 괄호 벗기고 x/X 마스킹만 제거, 한글 등 내용은 보존
#   3) 비언어 태그 b/ n/ l/ o/ u/ i/ (대소문자) → 위치·대소문자 불문 제거
#   4) 잔여 슬래시 제거 (간투어 등), 공백 정리
RE_DUAL    = re.compile(r"\(([^)]*)\)/\(([^)]*)\)")    # (표기)/(발음)
RE_DUAL2   = re.compile(r"\(([^()]*)\)\(([^()]*)\)")   # (표기)(발음)
RE_BRACKET = re.compile(r"\(\(([^)]*)\)\)")            # (( ... )) 마커 (빈/마스킹/한글 포함)
# 비언어 태그: 앞이 라틴이 '아닐' 때만 매칭 → 한글에 붙은 '그냥l/' '습니다n/' 도 잡힘.
# 정상 영단어(club/ 등)는 앞이 라틴이라 미매칭. 대문자/i 변종(O/ i/)까지 흡수.
RE_TAG     = re.compile(r"(?<![A-Za-z])[bnlouiBNLOUI]/")

def _unwrap_bracket(m):
    """(( inner )) → inner. 내부 x/X 마스킹 문자만 제거하고 한글 등은 보존."""
    inner = re.sub(r"[xX]+", " ", m.group(1))
    return " " + inner + " "

def clean_text(raw, dual_mode=DUAL_MODE):
    """text_norm: 이중전사 해소 + (( )) 마커/비언어 태그 제거. 구두점(.,?!)은 유지."""
    t = raw.strip()
    pick = (lambda m: m.group(1)) if dual_mode == "spell" else (lambda m: m.group(2))
    t = RE_DUAL.sub(pick, t)                 # 1) (A)/(B)
    t = RE_DUAL2.sub(pick, t)                # 1) (A)(B)
    t = RE_BRACKET.sub(_unwrap_bracket, t)   # 2) (( )) 괄호 벗김 + x 마스킹 제거, 한글 보존
    t = RE_TAG.sub(" ", t)                   # 3) b/ n/ l/ o/ u/ i/ (대소문자) 제거
    t = t.replace("/", "")                   # 4) 간투어 등 잔여 슬래시 제거(단어 유지)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,?!])", r"\1", t)     # 태그 제거로 생긴 '구두점 앞 공백' 정리 (습니다 . → 습니다.)
    return t
    # 주: 문장 첫머리 단독 'o'/'O' (o/ 의 슬래시 누락 변종, 전체 2건)는 의도적으로 두지 않고
    #     남긴다 — 2건 잡자고 일반 규칙을 넣으면 OK/O형 등 정상 표기 오탐 위험이 더 크다.

REQUIRED_KEYS = {"key", "audio", "duration", "text", "text_norm"}
def prune(rec):
    return {k: v for k, v in rec.items() if k in REQUIRED_KEYS or v is not None}

# ======================= 오디오 =======================
def to_wav16(src, dst):
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", str(TARGET_SR), "-ac", "1", str(dst)],
                   check=True, capture_output=True)

# ===================== manifest =====================
def build_rows(label_root, max_sessions):
    rows, bad_json, skipped = [], 0, 0
    jsons = sorted(label_root.rglob("S*.json"))
    if max_sessions:
        jsons = jsons[:max_sessions]
    for jp in jsons:
        cats = parse_categories(jp)
        if cats is None:
            skipped += 1
            continue
        dcat, jcat, session = cats
        try:
            ds = json.loads(jp.read_text(encoding="utf-8", errors="replace"))["dataSet"]
        except Exception:
            bad_json += 1
            continue
        ti  = ds.get("typeInfo", {})
        spk = {s["id"]: s for s in ti.get("speakers", [])}
        audio_base = label_to_audio_base(jp.parents[3])   # .../VL_D##
        for d in ds.get("dialogs", []):
            sid = d.get("speaker"); sm = spk.get(sid, {})
            ap  = d.get("audioPath", "")
            utt = Path(ap).stem
            rows.append({
                "uid": f"{dcat}_{jcat}_{session}_{utt}",   # 전역 유일 (충돌 불가)
                "session": session, "utt": utt,
                "speaker": sid, "spk_type": sm.get("type"),
                "gender": sm.get("gender"), "age": sm.get("age"),
                "tel_net": sm.get("telephone_network"),
                "domain": ti.get("category"),
                "duration": d.get("duration"),
                "raw": (d.get("text") or "").strip(),
                "src_wav": audio_base / ap,
            })
    return rows, bad_json, skipped, len(jsons)

def make_record(idx, r):
    rec = {
        "key": f"{CORPUS_ID}_{SPLIT}__{idx:05d}",
        "audio": f"audio/{r['uid']}.wav",
        "duration": r["duration"],
        "text": r["raw"],                      # 원본 전사 그대로 (원칙: text=원본)
        "text_norm": clean_text(r["raw"]),     # 전처리된 정답 (전처리는 여기서만)
        "age": r["age"],
        "gender": GENDER_MAP.get(r["gender"]),
        "spk_type": r["spk_type"],
        "domain": r["domain"],
        "tel_net": r["tel_net"],
        # 원본에 (( )) 마커(청취불가/마스킹)가 있던 발화 표시 → 채점 단계 선택적 제외용.
        # 마커 없으면 None → prune 이 키 자체를 제거.
        "has_mask": True if RE_BRACKET.search(r["raw"] or "") else None,
    }
    return prune(rec)

# ========================= 메인 =========================
def main():
    rows, bad_json, skipped, n_json = build_rows(LABEL_ROOT, MAX_SESSIONS)
    n_sess = len({r["uid"].rsplit("_", 1)[0] for r in rows})
    print(f"JSON {n_json:,}개 (파싱실패 {bad_json}, 경로형식 skip {skipped}) "
          f"→ 세션 {n_sess:,} / 발화 {len(rows):,}  "
          f"(MAX_SESSIONS={MAX_SESSIONS}, TARGET_SR={TARGET_SR})")

    # --- 게이트: uid 충돌 검사 ---
    dup = [(u, c) for u, c in Counter(r["uid"] for r in rows).items() if c > 1]
    print(f"[게이트] uid 중복: {len(dup)}   ← 0 이어야 함")
    if dup:
        for u, c in dup[:10]:
            print(f"   ⚠ {u} ×{c}")
        print("uid 충돌 → 경로 추출 점검 필요. 중단.")
        return

    records = [make_record(i, r) for i, r in enumerate(rows)]

    # --- 정규화 잔존 점검 (text_norm 기준; 구두점 유지가 정상) ---
    paren = [p for p in records if re.search(r"[()]", p["text_norm"])]
    bare  = [p for p in records if re.search(r"[A-Za-z0-9]", p["text_norm"])]
    empty = [p for p in records if p["text_norm"] == ""]
    print(f"\n[정규화 잔존: text_norm] 괄호 {len(paren)} / 영문·숫자 {len(bare)} "
          f"/ 빈 정답 {len(empty)} (총 {len(records):,})")
    for tag, lst in [("괄호", paren), ("영문·숫자", bare)]:
        for p in lst[:3]:
            print(f"   ({tag}) {p['text_norm'][:70]}")

    # --- 원본 오디오 사전 점검 ---
    miss_pre = [r["uid"] for r in rows if not r["src_wav"].exists()]
    print(f"\n[사전점검] 원본 wav 누락: {len(miss_pre)} / {len(rows)}")
    for u in miss_pre[:10]:
        print("   ", u)

    if DRY_RUN:
        print("\n--- 레코드 미리보기 2건 ---")
        for p in records[:2]:
            print(json.dumps(p, ensure_ascii=False, indent=2))
        print("\nDRY_RUN=True → 변환/쓰기 생략. 위 수치 확인 후 DRY_RUN=False 로 실행.")
        return

    # ----------------------- 실제 변환 -----------------------
    audio_out = OUT_DIR / "audio"
    if CLEAN_AUDIO:
        shutil.rmtree(audio_out, ignore_errors=True)
        print(f"[clean] 기존 audio/ 삭제: {audio_out}")
    audio_out.mkdir(parents=True, exist_ok=True)

    written = fail = miss = 0
    fails = []
    with open(OUT_DIR / "transcript.jsonl", "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            src = r["src_wav"]
            if not src.exists():
                miss += 1
                continue
            dst = audio_out / f"{r['uid']}.wav"
            try:
                if not dst.exists():
                    to_wav16(src, dst)         # 8k → 16k 업샘플
            except subprocess.CalledProcessError:
                fail += 1
                fails.append(r["uid"])
                continue
            rec = make_record(i, r)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  ... {written:,} 완료")

    print(f"\n완료: {written:,}건  (원본누락 {miss}, ffmpeg실패 {fail}) "
          f"→ {OUT_DIR / 'transcript.jsonl'}")
    if fails:
        print("실패 uid 일부:", fails[:10])

    # ----------------------- 사후 검증 -----------------------
    recs = [json.loads(l) for l in open(OUT_DIR / "transcript.jsonl", encoding="utf-8")]
    audios = [x["audio"] for x in recs]
    keys   = [x["key"] for x in recs]
    print(f"\n[검증] 총 {len(recs):,}건")
    print(f"[검증] audio 중복: {len(audios) - len(set(audios))}   ← 0")
    print(f"[검증] key 중복  : {len(keys) - len(set(keys))}   ← 0")
    print(f"[검증] domain 분포: {dict(Counter(x.get('domain') for x in recs))}")
    missing_files = [x['key'] for x in recs if not (OUT_DIR / x['audio']).exists()]
    print(f"[검증] 오디오 파일 누락: {len(missing_files)}")


if __name__ == "__main__":
    main()