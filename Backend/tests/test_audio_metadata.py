"""Focused tests for the shared audio-duration probe used by the custom
dataset service and its Speaker Verification repair path."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.services.audio_metadata import probe_audio_duration

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _write_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 0.5, freq: float = 440.0) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sample_rate, format="WAV")


def test_probe_returns_positive_duration_for_valid_wav(tmp_path):
    path = tmp_path / "clip.wav"
    _write_wav(path, seconds=0.5)

    result = probe_audio_duration(path)

    assert result is not None
    assert result.duration_seconds == pytest.approx(0.5, abs=0.05)
    assert result.sample_rate == 16000


def test_probe_returns_positive_duration_for_real_mp3():
    mp3_path = FIXTURES_DIR / "tiny_clip.mp3"
    if not mp3_path.exists():
        pytest.skip("tiny_clip.mp3 fixture not present")

    result = probe_audio_duration(mp3_path)

    assert result is not None
    assert result.duration_seconds > 0
    assert result.sample_rate > 0


def test_probe_returns_none_for_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.mp3"
    path.write_bytes(b"\x00\x01\x02\x03NOT REAL AUDIO DATA" * 50)

    result = probe_audio_duration(path)

    assert result is None


def test_probe_returns_none_for_missing_file(tmp_path):
    result = probe_audio_duration(tmp_path / "does_not_exist.wav")

    assert result is None
