"""Read-only discovery and listing for the ASVspoof 2019 LA demo subset.

Exposes the on-disk `asvspoof2019-la` recordings behind opaque, stable
recording ids (same pattern as the verification and diarization tasks).

GROUND-TRUTH SAFETY: an ASVspoof file id (`LA_E_2834763`) encodes nothing, so
it is safe to display -- but the protocol's `key` (bonafide/spoof) and
`system_id` (the attack, A07..A19) are exactly what this task asks the user to
judge. They are parsed only by `load_ground_truth()`, which is reserved for
offline evaluation, and must never reach a runtime response.

Expected on-disk layout (built by scripts/prepare_asvspoof_la_subset.py):

    Backend/data/deepfake/asvspoof2019_la/
        flac/LA_E_*.flac
        protocol.txt
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import settings

DATASET_ID = "asvspoof2019-la"
EXPECTED_RECORDING_COUNT = 200
SUPPORTED_AUDIO_EXTENSIONS = {".flac"}
PROTOCOL_FILENAME = "protocol.txt"


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
    # None when the header could not be read. The shared Audio Dataset table
    # renders this column, and without it every row reads "0.00s".
    duration_seconds: float | None = None


def _dataset_root() -> Path:
    return settings.asvspoof2019_la_dataset_dir


def _audio_dir() -> Path:
    return _dataset_root() / "flac"


def _recording_id_for(filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return f"rec_{digest[:16]}"


def _iter_audio_files(audio_dir: Path):
    if not audio_dir.is_dir():
        raise DatasetUnavailable(f"Deepfake demo dataset not found: {DATASET_ID}")

    for entry in audio_dir.iterdir():
        # Top level only: protocol.txt sits in the parent directory and is
        # offline-eval ground truth, so it is never reachable from here.
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        yield entry


def _read_duration(path: Path) -> float | None:
    """Clip length from the file header only -- no decoding, no audio read."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return round(info.frames / float(info.samplerate), 2)
    except Exception:
        # A listing must not fail because one file is unreadable.
        return None


def _discover(audio_dir: Path) -> list[RecordingInfo]:
    recordings: list[RecordingInfo] = []
    for entry in _iter_audio_files(audio_dir):
        recordings.append(
            RecordingInfo(
                recording_id=_recording_id_for(entry.name),
                display_filename=entry.name,
                extension=entry.suffix.lower(),
                size_bytes=entry.stat().st_size,
                duration_seconds=_read_duration(entry),
            )
        )
    recordings.sort(key=lambda recording: recording.display_filename)
    return recordings


def get_dataset_info() -> dict[str, object]:
    """Summarize the demo dataset without raising when it is absent."""

    try:
        recordings = _discover(_audio_dir())
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
    """List every recording in the demo subset. Raises if unavailable.

    Carries no label: see the module docstring.
    """

    return _discover(_audio_dir())


def get_recording(recording_id: str) -> RecordingInfo:
    """Look up a single recording by its opaque id."""

    for recording in list_recordings():
        if recording.recording_id == recording_id:
            return recording
    raise RecordingNotFound(f"Unknown recording id: {recording_id}")


def resolve_recording_path(recording_id: str) -> Path:
    """Resolve a safe recording id to its on-disk path. Internal use only --
    unknown or path-traversal ids simply miss and raise `RecordingNotFound`
    without touching the filesystem with untrusted input."""

    for entry in _iter_audio_files(_audio_dir()):
        if _recording_id_for(entry.name) == recording_id:
            return entry
    raise RecordingNotFound(f"Unknown recording id: {recording_id}")


def load_ground_truth() -> dict[str, tuple[str, str]]:
    """`{file_id: (system_id, key)}` parsed from protocol.txt.

    OFFLINE EVALUATION ONLY (Feature 1's score distribution / DET / EER). The
    `key` is the bona fide/spoof answer the workbench exists to let a user
    judge for themselves, so this must never be called from a route handler.

    Protocol lines are the ASVspoof 2019 CM format:
        SPEAKER_ID  FILE_ID  -  SYSTEM_ID  KEY
    """

    protocol_path = _dataset_root() / PROTOCOL_FILENAME
    if not protocol_path.is_file():
        raise DatasetUnavailable(
            f"Ground-truth protocol not found for {DATASET_ID}: {protocol_path}"
        )

    truth: dict[str, tuple[str, str]] = {}
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 5:
                continue
            _speaker_id, file_id, _unused, system_id, key = fields[:5]
            truth[file_id] = (system_id, key)
    return truth
