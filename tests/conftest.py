"""pytest 공통 fixture / 경로 설정."""

import sys
from pathlib import Path


# 프로젝트 루트를 import 경로에 추가 (editable install 안 했을 때 대비)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
