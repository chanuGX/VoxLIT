"""Read-only discovery and listing for the Speaker Verification demo dataset.

Exposes the on-disk `voxceleb1-indian-demo` recordings behind opaque, stable
recording ids. Original filenames are never returned: this dataset's
filenames (e.g. `id10018_01.wav`) encode a VoxCeleb speaker-id prefix that
groups recordings by speaker, and `metadata.csv` maps those ids to real
celebrity names — both are ground truth reserved for offline evaluation and
must never reach a runtime response.

Recordings additionally carry an anonymous `speaker_group_id` (e.g.
"speaker-group-3"), derived solely from the VoxCeleb speaker-id prefix
embedded in each filename. Unlike the prefix itself, this opaque group label
*is* allowed to reach runtime responses -- but only batch-analysis results,
where it is used as reporting-only ground truth (see `get_speaker_group_map`).
The raw prefix string, the real filename, and `metadata.csv` (which maps ids
to real celebrity names) are still never read for this purpose and never
reach any response, including the recording-browsing endpoints, which strip
`speaker_group_id` back out before returning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import torchaudio

from app.core.settings import settings

DATASET_ID = "voxceleb1-indian-demo"
EXPECTED_RECORDING_COUNT = 92
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

_SPEAKER_PREFIX_PATTERN = re.compile(r"^id\d+")


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
    duration_seconds: float | None
    speaker_group_id: str | None


def _dataset_dir() -> Path:
    return settings.speaker_verification_demo_dataset_dir


def _recording_id_for(filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return f"rec_{digest[:16]}"


def _speaker_prefix_for(filename: str) -> str | None:
    """Extract the VoxCeleb speaker-id prefix (e.g. "id10018" from
    "id10018_01.wav"), used only to derive an anonymous speaker-group id.
    Returns `None` on any non-matching filename rather than raising, so a
    stray file degrades gracefully instead of taking down dataset listing."""

    match = _SPEAKER_PREFIX_PATTERN.match(filename)
    return match.group(0) if match else None


def _duration_seconds(path: Path) -> float | None:
    """Header-only duration inspection (no full waveform decode). Returns
    `None` for any unreadable/corrupt/non-audio file rather than raising, so
    a single bad file never breaks the whole listing."""

    try:
        info = torchaudio.info(str(path))
        if info.sample_rate <= 0:
            return None
        return info.num_frames / info.sample_rate
    except Exception:
        return None


def _iter_audio_files(dataset_dir: Path):
    if not dataset_dir.is_dir():
        raise DatasetUnavailable(f"Speaker-verification demo dataset not found: {DATASET_ID}")

    for entry in dataset_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        yield entry


def _discover(dataset_dir: Path) -> list[RecordingInfo]:
    staged: list[tuple[str, str, int, float | None, str | None]] = []
    for entry in _iter_audio_files(dataset_dir):
        extension = entry.suffix.lower()
        recording_id = _recording_id_for(entry.name)
        staged.append(
            (
                recording_id,
                extension,
                entry.stat().st_size,
                _duration_seconds(entry),
                _speaker_prefix_for(entry.name),
            )
        )

    # Sort by opaque id rather than original filename so list order never
    # mirrors the speaker grouping encoded in the real filenames.
    staged.sort(key=lambda item: item[0])

    # Assign anonymous speaker-group ids in strict first-appearance order
    # over the now-sorted (by opaque id) list -- mirrors
    # `clustering.remap_labels_by_first_appearance`. Recordings whose
    # filename has no parseable speaker prefix get `speaker_group_id = None`.
    group_ids_by_prefix: dict[str, str] = {}
    recordings: list[RecordingInfo] = []
    for recording_id, extension, size_bytes, duration_seconds, prefix in staged:
        speaker_group_id: str | None = None
        if prefix is not None:
            if prefix not in group_ids_by_prefix:
                group_ids_by_prefix[prefix] = f"speaker-group-{len(group_ids_by_prefix) + 1}"
            speaker_group_id = group_ids_by_prefix[prefix]

        recordings.append(
            RecordingInfo(
                recording_id=recording_id,
                display_filename=f"{recording_id}{extension}",
                extension=extension,
                size_bytes=size_bytes,
                duration_seconds=duration_seconds,
                speaker_group_id=speaker_group_id,
            )
        )

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


def get_speaker_group_map() -> dict[str, str]:
    """Map `recording_id -> speaker_group_id` for the whole demo dataset.

    Recordings with no parseable speaker prefix are omitted, so
    `mapping.get(recording_id)` returning `None` uniformly covers both "not a
    demo recording id" and "demo recording with no parseable prefix" -- a
    caller building a ground-truth list for a batch never needs to
    distinguish the two. Never raises: if the dataset directory is
    unavailable, returns an empty mapping so a batch made entirely of
    non-demo (session-asset / custom-recording) ids is unaffected.
    """

    try:
        recordings = list_recordings()
    except DatasetUnavailable:
        return {}

    return {
        recording.recording_id: recording.speaker_group_id
        for recording in recordings
        if recording.speaker_group_id is not None
    }


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


def resolve_recording_path(recording_id: str) -> Path:
    """Resolve a safe recording id to its on-disk path.

    Used both internally (batch analysis) and by the
    `/dataset/recordings/{recording_id}/audio` route, which streams the
    resolved file's bytes but never returns this `Path`, the real filename,
    or any other filesystem detail in a response. Reuses the same id
    derivation as `get_recording`, so unknown or path-traversal ids simply
    miss and raise `RecordingNotFound` without touching the filesystem with
    untrusted input.
    """

    dataset_dir = _dataset_dir()
    for entry in _iter_audio_files(dataset_dir):
        if _recording_id_for(entry.name) == recording_id:
            return entry
    raise RecordingNotFound(f"Unknown recording id: {recording_id}")
