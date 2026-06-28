# 데이터 경로 & 전처리 과정

> 원천(RAW) → 전처리(SILVER) → 샘플링/구축(GOLD) 의 **현재 사실**.
> 벤치마크는 구축 완료(코드·디스크 확인), 학습셋은 폴더만 생성됨(전처리 TBD).
> 경로/규칙이 코드와 바뀌면 이 문서도 같이 바꾼다.

---

## 1. 데이터 경로 (디스크 실측 기준)

| 구분 | 경로 | 상태 |
|---|---|---|
| 원본(RAW) | `/data/ASR/RAW` | 24개 데이터셋 |
| 벤치마크 SILVER | `/data/ASR/BENCHMARK/SILVER` | 구축됨 (전량 전처리) |
| 벤치마크 GOLD | `/data/ASR/BENCHMARK/GOLD` | 구축됨 (SILVER 샘플링, 평가 기본 경로) |
| 학습 SILVER | `/data/ASR/TRAIN/SILVER` | **빈 폴더 (전처리 TBD)** |
| 학습 GOLD | `/data/ASR/TRAIN/GOLD` | **빈 폴더 (전처리 TBD)** |

```
RAW ──(전처리)──▶ SILVER ──(샘플링/구축)──▶ GOLD ──▶ 학습 / 평가
```

- 전처리는 **독립 과정**. 학습/평가 코드는 결과 JSONL **경로만** 입력받는다.
- 산출물은 발화 JSONL `transcript.jsonl` = `Sample` 스키마
  ([스키마](../GUIDELINE/스키마.md), 코드 `project/data/schema.py`).
  필수 키: `key`, `audio`(코퍼스 폴더 기준 상대경로), `duration`, `text`, `text_norm`.
  옵션: `age`, `gender`(`F`/`M`), `spk_type`, `domain` … (코드가 모르는 키는 무시).
- ⚠ 옛 `/data/ASR/BENCHMARK/SILVER/GOLD` 경로는 **비어있음/미사용**. 현재 GOLD는 `BENCHMARK/GOLD`.

---

## 2. 벤치마크 데이터 전처리 과정 (구축 완료)

### 2.1 RAW → SILVER (정제 / 포맷 통일)

데이터셋마다 라벨 형식·전사 방식이 달라 **데이터셋별 어댑터 스크립트**로 처리한다
(전역 정규식 금지). 공통적으로 수행하는 일반화된 작업:

1. **라벨 파싱 + 발화 단위 분리** — 세션 JSON / 발화별 txt 등 원천 포맷을 발화 레코드로.
2. **오디오 포맷 통일** — 16kHz / mono / wav. 8kHz 전화망 음원은 16kHz **업샘플**
   (음질 향상 목적 아님, 포맷 통일). `ffmpeg -ar 16000 -ac 1`.
3. **전사 정규화 (2단계)**
   - `text` : 이중전사(병기) 해소 + 비언어/특수 태그·마커 제거, **구두점 유지**.
     - 병기 괄호 `(표기)/(발음)` → 정책에 따라 발음형/표기형 택1.
       ★ **병기 순서가 MeetingASR만 반대** — 어댑터에서 순서 통일 필수
       ([preproc 이슈](../docs/preproc/02_이슈분석.md)).
     - 비언어 태그(`b/ n/ l/ o/ u/`), 간투어 슬래시, 자가수정/오발음 마커(`+`,`*`), PII 이름 마커(`@`) 등 제거.
   - `text_norm` : `text`에서 구두점 제거 + 공백 정리. (학습 타겟 + CER 비교 기준)
4. **화자/메타 채움** — `gender`(F/M), `age`, `spk_type`, `domain` 등. 값 없으면 키 자체 제거.
5. **길이 필터** — 너무 짧/긴 발화 분리(`filtered`), 무음 트리밍.
6. SILVER `transcript.jsonl` + `audio/` 작성. ⚠ 전사에 PII(이름 등) 잔존 가능 — 반출 주의.

### 2.2 SILVER → GOLD (샘플링)

코드: `benchmark_silver2gold.py` (yejin). 규칙:

- **상한 4000건 샘플링** — 코퍼스 발화 수가 `THRESHOLD`(4000) **미만이면 폴더 전체 하드카피**,
  **이상이면 4000건 랜덤 샘플링** 후 해당 음원+transcript만 GOLD로 복사.
- 랜덤 시드 고정 (`SEED=42`). 심볼릭 링크 아닌 **실제 파일 하드카피**.
- 실측: 대부분 4000건, 소스가 적은 코퍼스는 그 이하(NIKL_Dialect 500, Zeroth 457 등).

### 2.3 전처리 코드 위치 (작업자별, 리포 밖)

> `data_preproc/` 로 일원화 예정 ([docs/TODO.md](../docs/TODO.md) 참조).

- 기호(kiho) — `/home/kiho/workspace/TRAIN-ASR/notebooks/`
  (RAW→SILVER 어댑터: `RAW_to_SILVER/build_*_samples.py`)
- 예진(yejin) — `/home/yejin/workspace/TRAIN-ASR/notebooks/`
  (EDA + SILVER→GOLD: `EDA/benchmark_silver2gold.py`)

---

## 3. 학습 데이터 전처리 과정 (TBD — 폴더만 생성됨)

`/data/ASR/TRAIN/{SILVER,GOLD}` 은 **빈 폴더**. 아래는 확정 전 계획.

### 3.1 RAW → SILVER (TBD)

- **기본 방침: 벤치마크 RAW→SILVER 와 동일 어댑터/규칙 재사용** (2.1 그대로).
  같은 RAW 데이터셋·같은 `Sample` 스키마이므로 어댑터 공유가 자연스럽다.
- 벤치마크와 다른 점(확정 필요):
  - 출력 경로만 `/data/ASR/TRAIN/SILVER` 로.
  - 학습은 **양으로 승부** → 벤치마크용으로 떼어둔 평가 split 과 **음원 중복 배제**가 핵심
    (같은 발화가 학습/평가 양쪽에 들어가면 평가 무의미).
  - SILVER 전량(또는 high-confidence STT 포함)을 학습 후보로 둘지 등 필터 정책 TBD.

### 3.2 SILVER → GOLD (TBD)

- 벤치마크 GOLD(=평가 부분집합 샘플링)와 **목적이 다름**. 학습 GOLD는 "검수/정규화 완료 학습셋".
- 예상 작업:
  1. 여러 SILVER `transcript.jsonl` concat.
  2. 필터 (검수 전사 위주 / high-confidence 포함 정책 확정 TBD).
  3. 한국어 정규화 적용 → `text_norm` 채움.
  4. **train/val 분리** (시드 고정).
  5. 평가셋(벤치마크 GOLD/SILVER)과 **음원 겹침 없음** 검증.
  6. 셔플 후 `/data/ASR/TRAIN/GOLD/{train,val}.jsonl` 저장.
- 참고 설계: [docs/data/preprocessing.md](../docs/data/preprocessing.md) (권장 설계, 미구현).
