# Notebooks

탐색·분석·프로토타이핑용 Jupyter 노트북.

## 컨벤션

### 파일명
`<번호>_<설명>.ipynb` 형식. 번호는 노트북 실행 순서나 주제 묶음을 표현.

```
01_eda.ipynb              # 탐색적 데이터 분석
02_preprocessing.ipynb    # 전처리 검증
03_modeling.ipynb         # 모델링 실험
04_error_analysis.ipynb   # 에러 분석
```

### 셀 출력
**커밋 시 출력은 자동으로 제거됨** (`nbstripout` pre-commit hook).
출력이 필요한 결과물은 `reports/figures/`에 저장하거나 W&B로 로깅.

### 재사용 코드
노트북에서 같은 코드를 두 번 이상 쓰게 되면 `project/` 안의
적절한 모듈로 옮긴 뒤 import해서 사용. 노트북은 "탐색"용, 재사용 코드는 "라이브러리"용.

```python
# Bad: 노트북에 매번 같은 함수 정의
def normalize(x):
    ...

# Good: project 패키지로 옮긴 뒤 import
from project.data import normalize
```

### 의존성
노트북에서 추가 패키지가 필요하면 `pip install` 후 즉시 `requirements.txt`에 반영.
"노트북에서만 동작"하는 상태를 만들지 말 것.
