"""Reproducibility utilities — seed everything."""

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """모든 난수 생성기 시드를 고정.

    Args:
        seed: 시드 값.
        deterministic: True 인 경우 PyTorch cuDNN 결정론적 모드 활성화 (속도 손실 있음).

    Notes:
        PyTorch가 설치되어 있을 때만 torch 관련 시드를 설정. 미설치 시 조용히 건너뜀.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch 미설치 환경 — 무시
        pass
