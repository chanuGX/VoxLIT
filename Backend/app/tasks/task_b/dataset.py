"""Read-only discovery and listing for the AMI diarization demo dataset.

Exposes the on-disk `ami-subset` recordings behind opaque, stable recording
ids (same pattern as the verification task). Unlike VoxCeleb, AMI meeting
names (e.g. `ES2004a`) encode no ground truth, so the real filename is kept
as the display name. Ground-truth RTTMs live in the `rttm/` subfolder and are
reserved for offline evaluation — they must never reach a runtime response.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import settings

DATASET_ID = "ami-subset"
EXPECTED_RECORDING_COUNT = 3
SUPPORTED_AUDIO_EXTENSIONS = {".wav"}


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
    return settings.speaker_diarization_ami_dataset_dir


def _recording_id_for(filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return f"rec_{digest[:16]}"


def _iter_audio_files(dataset_dir: Path):
    if not dataset_dir.is_dir():
        raise DatasetUnavailable(f"Diarization demo dataset not found: {DATASET_ID}")

    for entry in dataset_dir.iterdir():
        # Top level only: the rttm/ subfolder is offline-eval ground truth
        # and is deliberately never listed.
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        yield entry


def _discover(dataset_dir: Path) -> list[RecordingInfo]:
    recordings: list[RecordingInfo] = []
    for entry in _iter_audio_files(dataset_dir):
        recordings.append(
            RecordingInfo(
                recording_id=_recording_id_for(entry.name),
                display_filename=entry.name,
                extension=entry.suffix.lower(),
                size_bytes=entry.stat().st_size,
            )
        )
    recordings.sort(key=lambda recording: recording.display_filename)
    return recordings


def get_dataset_info() -> dict[str, object]:
    """Summarize the demo dataset without raising when it is absent."""

    try:
        recordings = _discover(_dataset_dir())
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
    """Look up a single recording by its opaque id."""

    for recording in list_recordings():
        if recording.recording_id == recording_id:
            return recording
    raise RecordingNotFound(f"Unknown recording id: {recording_id}")


def resolve_recording_path(recording_id: str) -> Path:
    """Resolve a safe recording id to its on-disk path. Internal use only —
    unknown or path-traversal ids simply miss and raise `RecordingNotFound`
    without touching the filesystem with untrusted input."""

    for entry in _iter_audio_files(_dataset_dir()):
        if _recording_id_for(entry.name) == recording_id:
            return entry
    raise RecordingNotFound(f"Unknown recording id: {recording_id}")