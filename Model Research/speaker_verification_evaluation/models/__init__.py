from __future__ import annotations

from pathlib import Path


_SRC_MODELS = Path(__file__).resolve().parent.parent / "src" / "models"

if _SRC_MODELS.exists():
    __path__ = [str(_SRC_MODELS)]
else:
    __path__ = []
