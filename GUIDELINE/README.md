# GUIDELINE — 한국어 ASR 학습·평가 작업 가이드

음성인식기 학습과 평가를 위한 *최소 절차* 모음.

---

## 큰 그림

```
┌─ 1단계 (지금) ──────────────────────────────┐
│  data/Corpora_Meta_Table.xlsx               │  ← 오픈소스 코퍼스 정리
│  BENCHMARK/Benchmark_Meta_Table.xlsx        │  ← 벤치마크 초안 (오픈소스 + 커스텀)
│              │                              │
│              ▼                              │
│  BENCHMARK/<bench_id>/{audio/, transcript.jsonl}│
│              │                              │
│              ▼                              │
│  백본 모델로 평가 → BENCHMARK/results/...    │
└─────────────────────────────────────────────┘

┌─ 2단계 (TBD) ───────────────────────────────┐
│  data/SILVER/{train,val}.jsonl             │  ← 학습 데이터
│              │                              │
│              ▼                              │
│  학습 → outputs/<exp>/model.pt              │
│              │                              │
│              ▼                              │
│  학습된 모델로 1단계 벤치마크에 평가          │
└─────────────────────────────────────────────┘
```

---

## 읽는 순서

| # | 문서 | 무엇 | 상태 |
|---|---|---|---|
| 1 | [1_벤치마크구축.md](1_벤치마크구축.md) | 평가용 벤치마크 한 개 만들고, 백본 모델로 평가 | **지금 작업** |
| 2 | [2_모델학습.md](2_모델학습.md) | 모델 학습 (공통). 아키텍처별 → [Whisper](2_모델학습_whisper.md) / [SenseVoice](2_모델학습_sensevoice.md) | Whisper ✅ / SenseVoice 🚧 |
| 📋 | [스키마.md](스키마.md) | 발화 JSONL · 평가 yaml *공식 정의서* | 작업 중 *항상 참조* |

---

## 보조 자료 위치

| 무엇 | 어디 |
|---|---|
| 벤치마크 전처리 노트북 (템플릿) | [`../notebooks/00_build_benchmark.ipynb`](../notebooks/00_build_benchmark.ipynb) |
| Whisper 학습 노트북 | [`../notebooks/11_train_whisper.ipynb`](../notebooks/11_train_whisper.ipynb) |
| SenseVoice 학습 노트북 | [`../notebooks/10_train_sensevoice.ipynb`](../notebooks/10_train_sensevoice.ipynb) |
| 평가 yaml 모음 | [`../BENCHMARK/configs/eval/`](../BENCHMARK/configs/eval/) |
| 학습 yaml 베이스 | [`../configs/default.yaml`](../configs/default.yaml) |
| 코드 모듈 | `../project/` (`data/`, `evaluation/`, `training/`) |

---

## 공유 경로 (모두 동일)

| 무엇 | 경로 |
|---|---|
| 백본 모델 | `/data/FM/MODEL/{whisper-small, SenseVoiceSmall}` |
| 원본 데이터 | `/data/ASR-BENCHMARK/RAW/<source>/` |

각자 작업 영역은 `TRAIN-ASR/` 클론 안 (`BENCHMARK/`, `outputs/` 등).
