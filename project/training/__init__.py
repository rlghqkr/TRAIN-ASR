"""project.training — 학습 래퍼.

- `sensevoice.build_torchrun_command(cfg)` — FunASR `train_ds.py` 명령 빌드
- `whisper.build_trainer(cfg, ...)` — HF Trainer 빌드
"""

from . import sensevoice, whisper  # noqa: F401
