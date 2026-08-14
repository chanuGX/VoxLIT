"""Read-only discovery and listing for the Speaker Verification demo dataset.

Exposes the on-disk `voxceleb1-indian-demo` recordings behind opaque, stable
recording ids. Original filenames are never returned: this dataset's
filenames (e.g. `id10018_01.wav`) encode a VoxCeleb speaker-id prefix that
groups recordings by speaker, and `metadata.csv` maps those ids to real
celebrity names — both are ground truth reserved for offline evaluation and
must never reach a runtime response.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import settings

DATASET_ID = "voxceleb1-indian-demo"
EXPECTED_RECORDING_COUNT = 92
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


class DatasetUnavailable(RuntimeError):
    """Raised when the demo dataset directory is missing or unreadable."""


class RecordingNotFound(ValueError):
    """Raised when a recording id does not match a known recording."""


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    recording_id: str
    display_filename: str
    extension: str
    size_bytes: int


def _dataset_dir() -> Path:
    return settings.speaker_verification_demo_dataset_dir


def _recording_id_for(filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return f"rec_{digest[:16]}"


def _discover(dataset_dir: Path) -> list[RecordingInfo]:
    if not dataset_dir.is_dir():
        raise DatasetUnavailable(f"Speaker-verification demo dataset not found: {DATASET_ID}")

    recordings: list[RecordingInfo] = []
    for entry in dataset_dir.iterdir():
        if not entry.is_file():
            continue
        extension = entry.suffix.lower()
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        recording_id = _recording_id_for(entry.name)
        recordings.append(
            RecordingInfo(
                recording_id=recording_id,
                display_filename=f"{recording_id}{extension}",
                extension=extension,
                size_bytes=entry.stat().st_size,
            )
        )

    # Sort by opaque id rather than original filename so list order never
    # mirrors the speaker grouping encoded in the real filenames.
    recordings.sort(key=lambda recording: recording.recording_id)
    return recordings


def get_dataset_info() -> dict[str, object]:
    """Summarize the demo dataset without raising when it is absent."""

    dataset_dir = _dataset_dir()
    try:
        recordings = _discover(dataset_dir)
    except DatasetUnavailable:
        return {
            "dataset_id": DATASET_ID,
            "expected_recording_count": EXPECTED_RECORDING_COUNT,
            "total_recordings": 0,
            "audio_extensions": sorted(SUPPORTED_AUDIO_EXTENSIONS),
            "available": False,
        }

    return {
        "dataset_id": DATASET_ID,
        "expected_recording_count": EXPECTED_RECORDING_COUNT,
        "total_recordings": len(recordings),
        "audio_extensions": sorted(SUPPORTED_AUDIO_EXTENSIONS),
        "available": True,
    }


def list_recordings() -> list[RecordingInfo]:
    """List every recording in the demo dataset. Raises if unavailable."""

    return _discover(_dataset_dir())


def get_recording(recording_id: str) -> RecordingInfo:
    """Look up a single recording by its opaque id.

    Only ever checks membership in the map built from the real directory
    listing, so unknown or path-traversal ids simply miss and raise
    `RecordingNotFound` without touching the filesystem with untrusted input.
    """

    for recording in list_recordings():
        if recording.recording_id == recording_id:
            return recording
    raise RecordingNotFound(f"Unknown recording id: {recording_id}")
