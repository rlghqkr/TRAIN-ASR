# SOT (Source of Truth)

이 폴더는 **코드로 검증된 "현재 사실"과 확정된 설계 결정**만 담는다.
다른 docs(`study/`, `workflows/`, `training/` …)가 "어떻게 하는가(왓투두)"라면,
여기는 "지금 실제로 무엇이 참인가 / 왜 이렇게 정했는가"의 기준점이다.

## 규칙

- 여기 적힌 내용은 **코드와 일치**해야 한다. 코드가 바뀌면 이 문서도 같이 바뀐다.
- 아직 안 정해졌거나 코드에 없는 것은 적지 않는다 (희망사항·미구현 명세 금지).
- 다른 문서와 충돌하면 **이 폴더가 우선**한다.

## 문서

- [training-config-architecture.md](training-config-architecture.md) — 학습/평가 config 분리 구조, config 키별 실제 사용처, 환경·데이터 경로 규약
