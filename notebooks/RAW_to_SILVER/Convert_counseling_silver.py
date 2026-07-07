#!/usr/bin/env python3
"""AIHub CounselingSpeech (KtelSpeech) Validation → transcript.jsonl 벤치마크 변환 (수정본).

이전 버전 대비 바뀐 점 (디버깅으로 확정된 사항 반영):
  1) [치명적 버그 수정] 세션 일련번호 S******** 는 카테고리(D**/J**) '안에서만' 유일하다.
     이전 코드는 jp.stem 으로 세션만 떼어 D**/J** 를 버렸고, 그 결과 서로 다른 도메인의
     동명 세션(예: D60/J91/S00000365 와 D61/J92/S00000365)이 같은 출력 파일명으로
     충돌 → `if not dst.exists()` 때문에 뒤 발화가 앞 발화 오디오를 재사용 → 텍스트≠오디오.
     → 본 수정본은 uid = "<D**>_<J**>_<S********>_<utt>" 로 카테고리를 포함해 충돌을 원천 제거.
        카테고리/세션은 JSON 파일의 '실제 디스크 경로'에서 추출(내부 audioPath 문자열 불신).

  2) [매핑 — 확정 원칙] text 에는 무조건 '원본'이 들어가고, 전처리는 text_norm 에만 한다.
        text       = raw                    # 원본 전사 그대로 (태그·이중전사 포함)
        text_norm  = clean_text(raw)        # 전처리: 이중전사→발음형, 태그 제거, 구두점 '유지'
     (LowQuality 등 타 코퍼스와 동일 원칙·동일 text_norm 구두점 정책으로 통일.
      이전 수정본의 text=정리형은 이 원칙 위반이라 되돌림. normalize_text 제거.)

  3) [BOM 대응] 금융(D61) 배치 일부 JSON 63개가 UTF-8 BOM 으로 시작해 json.loads 가 거부 →
     전부 스킵됐었다. read_text(encoding="utf-8-sig") 로 읽어 63 세션 복구(도메인 균형 보강).

  4) [견고성] ffmpeg 변환을 발화 단위 try/except 로 감싸 한 파일 실패가 전체를 죽이지 않게 하고,
     원본 누락(miss)과 변환 실패(fail)를 분리 집계.

  5) [정규화] 비언어 태그(b/ n/ l/ o/ u/, 전부 소문자) 제거 정규식을 한글에 붙은 경우
     (예: '그래n/')까지 잡도록 (?<![A-Za-z])[bnlou]/ 로 보강. @(이름)/+( 자가수정)/*(불명확)
     은 CounselingSpeech 고유 마커로 제거. (( )) 마커·대문자 태그는 이 코퍼스에 없어 미적용.

DRY_RUN=True 로 먼저 uid 충돌 0 / 정규화 잔존 / 원본 누락을 확인한 뒤, False 로 실제 변환.
출력:
  OUT_DIR/transcript.jsonl
  OUT_DIR/audio/<D**>_<J**>_<S********>_<utt>.wav   (8kHz → 16kHz 업샘플, mono)
"""
from pathlib import Path
from collections import Counter
import json
import re
import subprocess
import shutil

try:
    import soundfile as sf
except ImportError:
    sf = None

# ============================ 설정 ============================
LABEL_ROOT = Path(
    "/data/ASR/RAW/AIHub_CounselingSpeech/012.상담_음성_데이터"
    "/01.데이터/2.Validation/라벨링데이터_1129_add"
)
OUT_DIR   = Path("/data/ASR/BENCHMARK/SILVER/AIHub_CounselingSpeech")  # 기존 위치 재생성
CORPUS_ID = "AIHub_CounselingSpeech"
SPLIT     = "valid"
TARGET_SR = 16000          # 8kHz 전화망 → 16kHz (음질 향상 아님, 포맷 통일)
DUAL_MODE = "pron"         # 이중전사: 'pron'=발음형 / 'spell'=표기형
MAX_SESSIONS = None        # 1차 확인용으로 정수 지정 가능. 전체 변환 시 None
DRY_RUN   = False           # 먼저 True 로 검증 → False 로 변환
CLEAN_AUDIO = True        # True 면 변환 전 기존 audio/ 를 통째로 삭제(충돌 잔재 제거)

GENDER_MAP = {"여": "F", "남": "M"}

# 경로 계층 검증용 (.../<Dxx>/<Jxx>/<Sxxxxxxxx>/<Sxxxxxxxx>.json)
RE_DCAT = re.compile(r"^D\d{2}$")
RE_JCAT = re.compile(r"^J\d{2}$")
RE_SESS = re.compile(r"^S\d{8}$")

# ========================= 경로 매핑 =========================
def label_to_audio(p: Path) -> Path:
    """라벨링데이터 경로 → 원천데이터(오디오) 경로."""
    return Path(str(p).replace("라벨링데이터", "원천데이터").replace("_label_", "_wav_"))

def parse_categories(jp: Path):
    """JSON 파일의 디스크 경로에서 (D**, J**, S********) 추출. 형식 불일치 시 None."""
    try:
        session = jp.parent.name
        jcat    = jp.parent.parent.name
        dcat    = jp.parent.parent.parent.name
    except Exception:
        return None
    if not (RE_DCAT.match(dcat) and RE_JCAT.match(jcat) and RE_SESS.match(session)):
        return None
    return dcat, jcat, session

# ========================== 정규화 ==========================
RE_DUAL  = re.compile(r"\(([^)]*)\)/\(([^)]*)\)")     # (표기)/(발음)
RE_DUAL2 = re.compile(r"\(([^()]*)\)\(([^()]*)\)")    # (표기)(발음)  예: (10)(십)
# 비언어 태그 b/ n/ l/ o/ u/ : 앞이 라틴문자가 '아닐' 때 매칭 → '그래n/' 같이 한글에 붙어도 잡힘
# (CounselingSpeech 태그는 전부 소문자 — 대문자/i 변종 없음 확인됨)
RE_NOISE = re.compile(r"(?<![A-Za-z])[bnlou]/")

def clean_text(raw: str, dual_mode: str = DUAL_MODE) -> str:
    """text_norm: 이중전사 해소 + 태그/마커 제거. 구두점(.,?!)은 유지 (LowQuality와 통일)."""
    t = raw.strip()
    pick = (lambda m: m.group(1)) if dual_mode == "spell" else (lambda m: m.group(2))
    t = RE_DUAL.sub(pick, t)        # (A)/(B) 먼저
    t = RE_DUAL2.sub(pick, t)       # (A)(B) 그다음
    t = RE_NOISE.sub(" ", t)        # 비언어 태그 b/ n/ l/ o/ u/ 삭제
    t = t.replace("@", "")          # PII 이름 마커 기호만 제거(발화 단어는 유지) — CounselingSpeech 고유
    t = t.replace("/", "")          # 간투어 슬래시('어/' '막/')의 '/' 제거
    t = t.replace("+", " ").replace("*", " ")   # 자가수정(+)/불명확(*) 마커 제거 — CounselingSpeech 고유
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,?!])", r"\1", t)        # 태그 제거로 생긴 '구두점 앞 공백' 정리
    return t

REQUIRED_KEYS = {"key", "audio", "duration", "text", "text_norm"}
def prune(rec: dict) -> dict:
    """필수 키 외 옵션 필드는 값이 None이면 키 자체를 제거."""
    return {k: v for k, v in rec.items() if k in REQUIRED_KEYS or v is not None}

# =========================== 오디오 ==========================
def wav_duration(p: Path) -> float:
    info = sf.info(str(p))
    return round(info.frames / info.samplerate, 3)

def to_wav16(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", str(TARGET_SR), "-ac", "1", str(dst)],
        check=True, capture_output=True,
    )

# ========================= manifest =========================
def build_rows(label_root: Path, max_sessions=None):
    """발화 단위 row 목록 생성. uid 는 카테고리 포함 → 전역 유일."""
    rows, skipped = [], 0
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
            ds = json.loads(jp.read_text(encoding="utf-8-sig", errors="replace"))["dataSet"]
        except Exception:
            skipped += 1
            continue
        ti  = ds.get("typeInfo", {})
        spk = {s["id"]: s for s in ti.get("speakers", [])}
        domain    = ti.get("category")
        audio_dir = label_to_audio(jp.parent)
        for d in ds.get("dialogs", []):
            try:
                txt_name = Path(d["textPath"]).name
                wav_name = Path(d["audioPath"]).name
            except (KeyError, TypeError):
                skipped += 1
                continue
            utt = Path(wav_name).stem            # 0001
            tf = jp.parent / txt_name
            if not tf.exists():
                skipped += 1
                continue
            sm = spk.get(d.get("speaker"), {})
            rows.append({
                "uid": f"{dcat}_{jcat}_{session}_{utt}",   # 전역 유일 (충돌 불가)
                "domain": domain,
                "spk_type": sm.get("type"),
                "gender": sm.get("gender"),
                "age": sm.get("age"),
                "raw": tf.read_text(encoding="utf-8-sig", errors="replace").strip(),
                "src_wav": audio_dir / wav_name,
                # 참고용(필요 시 디버깅)
                "txt_stem": Path(txt_name).stem,
                "wav_stem": utt,
            })
    return rows, skipped

def make_record(idx: int, r: dict) -> dict:
    rec = {
        "key": f"{CORPUS_ID}_{SPLIT}__{idx:05d}",
        "audio": f"audio/{r['uid']}.wav",
        "duration": None,
        "text": r["raw"],                    # 원본 전사 그대로 (원칙: text=원본)
        "text_norm": clean_text(r["raw"]),   # 전처리된 정답 (전처리는 여기서만, 구두점 유지)
        "age": r["age"],
        "gender": GENDER_MAP.get(r["gender"]),
        "spk_type": r["spk_type"],
        "domain": r["domain"],
    }
    return prune(rec)

# =========================== 메인 ============================
def main():
    if sf is None:
        print("⚠ soundfile 필요: pip install soundfile")
        return

    rows, skipped = build_rows(LABEL_ROOT, MAX_SESSIONS)
    n_sessions = len({r["uid"].rsplit("_", 1)[0] for r in rows})
    print(f"세션 {n_sessions:,} / 발화 {len(rows):,}  (skip {skipped}, "
          f"MAX_SESSIONS={MAX_SESSIONS}, TARGET_SR={TARGET_SR})")

    # --- 게이트 1: uid 충돌 검사 (이번 수정의 핵심) ---
    uid_counts = Counter(r["uid"] for r in rows)
    dup = [(u, c) for u, c in uid_counts.items() if c > 1]
    print(f"[게이트] uid 중복: {len(dup)}   ← 0 이어야 함")
    if dup:
        for u, c in dup[:10]:
            print(f"   ⚠ {u} ×{c}")
        print("uid 충돌 발견 → 경로 추출 로직 점검 필요. 중단.")
        return

    # --- 게이트 2: stem 정합(텍스트/오디오 발화번호 일치) 참고 검사 ---
    stem_mismatch = [r["uid"] for r in rows if r["txt_stem"] != r["wav_stem"]]
    if stem_mismatch:
        print(f"[게이트] txt/wav stem 불일치: {len(stem_mismatch)} (uid는 wav 기준)")
        for u in stem_mismatch[:5]:
            print("   ", u)

    records = [make_record(i, r) for i, r in enumerate(rows)]

    # --- 정규화 잔존 점검 (text_norm 기준) ---
    paren = [p for p in records if re.search(r"[()]", p["text_norm"])]
    bare  = [p for p in records if re.search(r"[A-Za-z0-9]", p["text_norm"])]
    empty = [p for p in records if p["text_norm"] == ""]
    print(f"\n[정규화 잔존: text_norm] 괄호 {len(paren)} / 영문·숫자 {len(bare)} "
          f"/ 빈 정답 {len(empty)} (총 {len(records)})")
    for tag, lst in [("괄호", paren), ("영문·숫자", bare)]:
        for p in lst[:3]:
            print(f"   ({tag}) {p['text_norm'][:70]}")

    # --- 원본 오디오 존재 사전 점검 ---
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
                dur = wav_duration(dst)
            except subprocess.CalledProcessError:
                fail += 1
                fails.append(r["uid"])
                continue
            rec = make_record(i, r)
            rec["duration"] = dur
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 2000 == 0:
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