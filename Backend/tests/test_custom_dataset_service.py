"""Focused tests for the hardened Custom Dataset Manager: session
authorization (the IDOR fix), path-traversal rejection, upload limits,
per-dataset lock scope under concurrency, and atomic metadata writes.

All storage is isolated to `tmp_path` -- no test ever touches the real
`Backend/uploads/sessions` directory or an existing dataset.
"""

from __future__ import annotations

import io
import secrets
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from starlette.datastructures import UploadFile

from app.core.session import InvalidSessionId, validate_session_id
from app.services import custom_dataset_service as cds
from app.services import dataset_service

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _valid_sid() -> str:
    return secrets.token_hex(16)  # 32 lowercase hex chars, matches ensure_session()'s .hex format


def _wav_bytes(*, sample_rate: int = 16000, seconds: float = 0.2, freq: float = 440.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _mp3_bytes() -> bytes:
    return (FIXTURES_DIR / "tiny_clip.mp3").read_bytes()


def _corrupt_bytes() -> bytes:
    return b"\x00\x01\x02\x03NOT REAL AUDIO DATA" * 50


def _make_upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch, tmp_path):
    """Every test gets its own storage root so nothing touches the real
    Backend/uploads/sessions directory -- covers both direct
    CustomDatasetManager(base_dir=...) construction and the module-level
    default path used by get_custom_dataset_manager()/dataset_service.py."""
    root = tmp_path / "sessions"
    monkeypatch.setattr(cds, "SESSIONS_BASE_DIR", root)
    return root


# ---------------------------------------------------------------------------
# validate_session_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_sid",
    [None, "", "short", "G" * 32, "-" * 32, "{" + "a" * 32 + "}", secrets.token_hex(16).upper()],
)
def test_validate_session_id_rejects_malformed(bad_sid):
    with pytest.raises(InvalidSessionId):
        validate_session_id(bad_sid)


def test_validate_session_id_accepts_canonical_shape():
    sid = _valid_sid()
    assert validate_session_id(sid) == sid


# ---------------------------------------------------------------------------
# Happy path: create / list / upload / resolve / delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_list_upload_resolve_delete_roundtrip(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)

    created = manager.create_dataset("My Dataset")
    assert created["dataset_name"] == "My Dataset"
    assert created["files"] == []

    listed = manager.list_datasets()
    assert [d["dataset_name"] for d in listed] == ["My Dataset"]

    upload = _make_upload(_wav_bytes(), "clip.wav")
    file_metadata = await manager.add_file_to_dataset_stream("My Dataset", "clip.wav", upload)
    assert file_metadata["filename"] == "clip.wav"
    assert file_metadata["original_filename"] == "clip.wav"
    assert file_metadata["size"] > 0
    assert file_metadata["duration"] > 0

    metadata = manager.get_dataset_metadata("My Dataset")
    assert metadata["total_files"] == 1
    assert metadata["files"][0]["filename"] == "clip.wav"

    resolved = manager.resolve_file_path("My Dataset", "clip.wav")
    assert resolved.is_file()
    assert resolved.stat().st_size > 0

    assert manager.delete_dataset("My Dataset") is True
    assert manager.get_dataset_metadata("My Dataset") is None
    assert manager.delete_dataset("My Dataset") is False


@pytest.mark.asyncio
async def test_duplicate_filename_is_deduplicated_not_overwritten(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")

    first = await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))
    second = await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))

    assert first["filename"] == "clip.wav"
    assert second["filename"] == "clip_1.wav"
    metadata = manager.get_dataset_metadata("d1")
    assert metadata["total_files"] == 2
    assert {f["filename"] for f in metadata["files"]} == {"clip.wav", "clip_1.wav"}


def test_resolve_missing_file_raises_file_not_found(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    with pytest.raises(FileNotFoundError):
        manager.resolve_file_path("d1", "nope.wav")


# ---------------------------------------------------------------------------
# Path traversal rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "../evil",
        "..",
        "a/b",
        "a\\b",
        "a\x00b",
        "con:trol",
        "a" * 101,
        "%2e%2e%2f",  # rejected because ':', '%' etc. are outside the allowed charset
    ],
)
def test_dataset_name_traversal_rejected(tmp_path, bad_name):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    with pytest.raises(cds.InvalidDatasetName):
        manager.create_dataset(bad_name)
    # Nothing must have been created on disk for a rejected name.
    assert list(manager.datasets_dir.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_filename",
    [
        "",
        "../../etc/passwd.wav",
        "..\\..\\windows\\system32\\evil.wav",
        "a/b.wav",
        "a\\b.wav",
        "clip.exe",
        "clip",
        "clip\x00.wav",
    ],
)
async def test_filename_traversal_rejected_on_upload(tmp_path, bad_filename):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    dataset_dir = manager.datasets_dir / "d1"

    with pytest.raises(cds.InvalidFilename):
        await manager.add_file_to_dataset_stream("d1", bad_filename, _make_upload(_wav_bytes(), bad_filename))

    # No file (temp or final) must have been written for a rejected name.
    assert [p.name for p in dataset_dir.iterdir()] == ["dataset_metadata.json"]


@pytest.mark.parametrize(
    "bad_filename",
    ["../../etc/passwd", "a/b.wav", "a\\b.wav", "..", ""],
)
def test_filename_traversal_rejected_on_resolve(tmp_path, bad_filename):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    with pytest.raises((cds.InvalidFilename, FileNotFoundError)):
        manager.resolve_file_path("d1", bad_filename)


def test_delete_dataset_cannot_escape_session_root(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    # A traversal-shaped name is rejected before delete_dataset ever computes
    # a candidate path to rmtree.
    with pytest.raises(cds.InvalidDatasetName):
        manager.delete_dataset("../d1")
    # The legitimately created dataset must still exist, untouched.
    assert manager.get_dataset_metadata("d1") is not None


# ---------------------------------------------------------------------------
# Upload limits: oversized file, per-dataset capacity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_file_rejected_and_leaves_no_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cds, "MAX_FILE_BYTES", 1000)
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    dataset_dir = manager.datasets_dir / "d1"

    oversized = b"R" * 5000
    with pytest.raises(cds.FileTooLarge):
        await manager.add_file_to_dataset_stream("d1", "big.wav", _make_upload(oversized, "big.wav"))

    # No temp file, no partial/final file -- only the metadata file remains.
    assert [p.name for p in dataset_dir.iterdir()] == ["dataset_metadata.json"]
    assert manager.get_dataset_metadata("d1")["total_files"] == 0


@pytest.mark.asyncio
async def test_max_files_per_dataset_enforced_and_dataset_stays_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(cds, "MAX_FILES_PER_DATASET", 1)
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")

    await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))

    with pytest.raises(cds.DatasetTooLarge):
        await manager.add_file_to_dataset_stream("d1", "clip2.wav", _make_upload(_wav_bytes(), "clip2.wav"))

    # The dataset remains fully readable/listable at the limit -- only
    # further writes are rejected.
    metadata = manager.get_dataset_metadata("d1")
    assert metadata["total_files"] == 1
    assert manager.resolve_file_path("d1", "clip.wav").is_file()


# ---------------------------------------------------------------------------
# Concurrency: lock scope covers capacity check + name allocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_uploads_same_filename_do_not_collide(tmp_path):
    import asyncio

    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")

    results = await asyncio.gather(
        manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(freq=440.0), "clip.wav")),
        manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(freq=660.0), "clip.wav")),
    )

    stored_names = {r["filename"] for r in results}
    assert stored_names == {"clip.wav", "clip_1.wav"}
    metadata = manager.get_dataset_metadata("d1")
    assert metadata["total_files"] == 2
    assert {f["filename"] for f in metadata["files"]} == stored_names


@pytest.mark.asyncio
async def test_concurrent_uploads_never_both_pass_stale_capacity_check(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(cds, "MAX_FILES_PER_DATASET", 1)
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")

    results = await asyncio.gather(
        manager.add_file_to_dataset_stream("d1", "a.wav", _make_upload(_wav_bytes(), "a.wav")),
        manager.add_file_to_dataset_stream("d1", "b.wav", _make_upload(_wav_bytes(), "b.wav")),
        return_exceptions=True,
    )

    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], cds.DatasetTooLarge)
    assert manager.get_dataset_metadata("d1")["total_files"] == 1


# ---------------------------------------------------------------------------
# Atomic metadata write: a failed replace never orphans the audio file or
# corrupts the previously valid metadata.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_write_failure_cleans_up_orphaned_audio_and_preserves_metadata(tmp_path, monkeypatch):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    dataset_dir = manager.datasets_dir / "d1"
    metadata_file = dataset_dir / "dataset_metadata.json"
    original_bytes = metadata_file.read_bytes()

    def failing_write(dataset_dir_arg, metadata):
        raise RuntimeError("simulated metadata write failure")

    monkeypatch.setattr(cds.CustomDatasetManager, "_write_metadata_atomic", staticmethod(failing_write))

    with pytest.raises(RuntimeError):
        await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))

    assert metadata_file.read_bytes() == original_bytes
    assert [p.name for p in dataset_dir.iterdir()] == ["dataset_metadata.json"]


# ---------------------------------------------------------------------------
# dataset_service.py: the IDOR fix (session authorization for custom datasets)
# ---------------------------------------------------------------------------


def _create_dataset_with_file(sid: str) -> str:
    manager = cds.CustomDatasetManager(sid)
    manager.create_dataset("d1")
    return cds.format_custom_dataset_name(sid, "d1")


@pytest.mark.asyncio
async def test_load_metadata_owner_can_access_own_dataset(isolated_storage):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid)
    manager.create_dataset("d1")
    await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))
    dataset_id = cds.format_custom_dataset_name(sid, "d1")

    rows = dataset_service.load_metadata(dataset_id, session_id=sid)
    assert len(rows) == 1
    assert rows[0]["filename"] == "clip.wav"


def test_load_metadata_rejects_cross_session_access(isolated_storage):
    sid_owner = _valid_sid()
    sid_attacker = _valid_sid()
    dataset_id = _create_dataset_with_file(sid_owner)

    with pytest.raises(ValueError):
        dataset_service.load_metadata(dataset_id, session_id=sid_attacker)


def test_load_metadata_rejects_malformed_session(isolated_storage):
    sid_owner = _valid_sid()
    dataset_id = _create_dataset_with_file(sid_owner)

    with pytest.raises(ValueError):
        dataset_service.load_metadata(dataset_id, session_id="not-a-real-session-id")

    with pytest.raises(ValueError):
        dataset_service.load_metadata(dataset_id, session_id=None)


@pytest.mark.asyncio
async def test_resolve_file_owner_can_access_own_dataset(isolated_storage):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid)
    manager.create_dataset("d1")
    await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))
    dataset_id = cds.format_custom_dataset_name(sid, "d1")

    resolved = dataset_service.resolve_file(dataset_id, "clip.wav", session_id=sid)
    assert resolved.is_file()


@pytest.mark.asyncio
async def test_resolve_file_rejects_cross_session_access(isolated_storage):
    sid_owner = _valid_sid()
    sid_attacker = _valid_sid()
    manager = cds.CustomDatasetManager(sid_owner)
    manager.create_dataset("d1")
    await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))
    dataset_id = cds.format_custom_dataset_name(sid_owner, "d1")

    with pytest.raises(FileNotFoundError):
        dataset_service.resolve_file(dataset_id, "clip.wav", session_id=sid_attacker)


def test_resolve_file_rejects_missing_session(isolated_storage):
    sid_owner = _valid_sid()
    dataset_id = _create_dataset_with_file(sid_owner)

    with pytest.raises(FileNotFoundError):
        dataset_service.resolve_file(dataset_id, "clip.wav", session_id=None)


def test_load_metadata_builtin_dataset_ignores_session(isolated_storage):
    # Built-in datasets never require or check a session -- this is a
    # before/after parity guard that the custom-dataset hardening above
    # doesn't touch this path.
    rows = dataset_service.load_metadata("common-voice", session_id=None)
    assert isinstance(rows, list)
    assert len(rows) > 0


# ---------------------------------------------------------------------------
# Duration extraction: fresh uploads, extraction failure, repair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_wav_upload_returns_positive_duration(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")

    file_metadata = await manager.add_file_to_dataset_stream(
        "d1", "clip.wav", _make_upload(_wav_bytes(seconds=0.5), "clip.wav")
    )

    assert file_metadata["duration"] is not None
    assert file_metadata["duration"] > 0
    assert file_metadata["sample_rate"] is not None


@pytest.mark.asyncio
async def test_fresh_mp3_upload_returns_positive_duration(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")

    file_metadata = await manager.add_file_to_dataset_stream(
        "d1", "clip.mp3", _make_upload(_mp3_bytes(), "clip.mp3")
    )

    assert file_metadata["duration"] is not None
    assert file_metadata["duration"] > 0
    assert file_metadata["sample_rate"] is not None


@pytest.mark.asyncio
async def test_extraction_failure_stores_null_not_zero(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")

    file_metadata = await manager.add_file_to_dataset_stream(
        "d1", "corrupt.wav", _make_upload(_corrupt_bytes(), "corrupt.wav")
    )

    assert file_metadata["duration"] is None
    assert file_metadata["sample_rate"] is None

    metadata = manager.get_dataset_metadata("d1")
    assert metadata["files"][0]["duration"] is None


@pytest.mark.asyncio
async def test_repair_missing_durations_fixes_zero_duration_entry(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(seconds=0.5), "clip.wav"))

    # Simulate a historically-broken entry the way the old librosa fallback
    # used to persist it, bypassing the (now-fixed) upload path entirely.
    dataset_dir = tmp_path / sid / "datasets" / "d1"
    metadata = cds.CustomDatasetManager._read_metadata(dataset_dir)
    metadata["files"][0]["duration"] = 0.0
    metadata["files"][0]["sample_rate"] = 0
    cds.CustomDatasetManager._write_metadata_atomic(dataset_dir, metadata)

    repaired = await manager.repair_missing_durations("d1")

    assert repaired["files"][0]["duration"] > 0
    assert repaired["files"][0]["sample_rate"] > 0
    persisted = manager.get_dataset_metadata("d1")
    assert persisted["files"][0]["duration"] > 0


@pytest.mark.asyncio
async def test_repair_missing_durations_leaves_unresolvable_entry_null(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))

    dataset_dir = tmp_path / sid / "datasets" / "d1"
    metadata = cds.CustomDatasetManager._read_metadata(dataset_dir)
    metadata["files"][0]["duration"] = None
    metadata["files"][0]["filename"] = "gone.wav"  # no longer resolves on disk
    cds.CustomDatasetManager._write_metadata_atomic(dataset_dir, metadata)

    repaired = await manager.repair_missing_durations("d1")

    assert repaired["files"][0]["duration"] is None


@pytest.mark.asyncio
async def test_repair_missing_durations_only_writes_when_something_changed(tmp_path):
    sid = _valid_sid()
    manager = cds.CustomDatasetManager(sid, base_dir=tmp_path)
    manager.create_dataset("d1")
    await manager.add_file_to_dataset_stream("d1", "clip.wav", _make_upload(_wav_bytes(), "clip.wav"))

    dataset_dir = tmp_path / sid / "datasets" / "d1"
    metadata_file = dataset_dir / "dataset_metadata.json"
    mtime_before = metadata_file.stat().st_mtime_ns

    await manager.repair_missing_durations("d1")

    assert metadata_file.stat().st_mtime_ns == mtime_before
