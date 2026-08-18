"""Shared audio-duration probing, reused by the custom dataset service and
its Speaker Verification repair path.

Two-stage probe: a fast header-only read first (`torchaudio.info`, the same
technique already used for the demo dataset in
`app/tasks/verification/dataset.py`), then a fall back to a full decode via
`torchaudio.load` -- the same decoder Speaker Verification already relies on
to successfully load these exact files for embedding extraction. Never
raises; returns `None` when both stages fail so callers can persist/report
"unknown" instead of a false zero duration.

Both stages do blocking file I/O (and possibly a full waveform decode), so
callers on an event loop must invoke `probe_audio_duration` via
`starlette.concurrency.run_in_threadpool` rather than awaiting it directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torchaudio

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioProbeResult:
    duration_seconds: float
    sample_rate: int


def probe_audio_duration(path: Path) -> AudioProbeResult | None:
    """Best-effort duration probe for WAV/MP3/FLAC/M4A. Returns `None`, never
    a false `0.0`, if neither stage can read the file."""

    try:
        info = torchaudio.info(str(path))
        if info.sample_rate > 0 and info.num_frames > 0:
            return AudioProbeResult(info.num_frames / info.sample_rate, info.sample_rate)
        logger.warning(f"Header-only duration probe returned no frames for {path.name}")
    except Exception as e:
        logger.warning(f"Header-only duration probe failed for {path.name}: {e}")

    try:
        waveform, sample_rate = torchaudio.load(str(path))
        if sample_rate > 0 and waveform.numel() > 0:
            return AudioProbeResult(waveform.shape[-1] / sample_rate, sample_rate)
        logger.warning(f"Full-decode duration probe returned no samples for {path.name}")
    except Exception as e:
        logger.warning(f"Full-decode duration probe failed for {path.name}: {e}")

    return None
