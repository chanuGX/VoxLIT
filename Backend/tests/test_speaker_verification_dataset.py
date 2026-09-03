"""Focused tests for demo-dataset discovery and listing (Commit 2).

All tests build a fake `vox_indian_demo_92`-shaped directory under `tmp_path`
and point `settings.SPEAKER_VERIFICATION_DATASET_ROOT` at it via monkeypatch —
never the real, gitignored dataset — mirroring the pattern used in
`test_speaker_verification_model_storage.py` for model directories.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from app.core.settings import settings
from app.tasks.verification import dataset

REAL_DATASET_DIR = settings.speaker_verification_demo_dataset_dir

FAKE_FILES = {
    "id10018_01.wav": "Akshay_Kumar",
    "id10018_02.wav": "Akshay_Kumar",
    "id10324_01.wav": "Freida_Pinto",
}


def _build_fake_dataset(root):
    dataset_dir = root / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    for filename in FAKE_FILES:
        (dataset_dir / filename).write_bytes(b"RIFF-fake-audio")
    (dataset_dir / "metadata.csv").write_text(
        "file_name,speaker_name\n"
        + "\n".join(f"{name},{speaker}" for name, speaker in FAKE_FILES.items())
    )
    (dataset_dir / "notes.txt").write_text("not audio")
    return dataset_dir


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    return _build_fake_dataset(tmp_path)


def _build_dataset_with_filenames(root, filenames):
    """Bare-bones fake dataset (no metadata.csv) for tests that need a
    filename set FAKE_FILES can't provide -- e.g. a non-matching filename or
    a specific opaque-id sort order. Kept separate from `_build_fake_dataset`
    so the many tests relying on the fixed `FAKE_FILES` shape are unaffected.
    """

    dataset_dir = root / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    for filename in filenames:
        (dataset_dir / filename).write_bytes(b"RIFF-fake-audio")
    return dataset_dir


def _assert_no_ground_truth_leak(payload) -> None:
    serialized = json.dumps(payload)
    for filename, speaker in FAKE_FILES.items():
        assert filename not in serialized
        assert speaker not in serialized
    assert "metadata.csv" not in serialized
    assert "vox_indian_demo_92" not in serialized


def test_dataset_id_is_exact():
    assert dataset.DATASET_ID == "voxceleb1-indian-demo"


def test_get_dataset_info_reports_counts_and_extensions(fake_dataset_dir):
    info = dataset.get_dataset_info()

    assert info["dataset_id"] == "voxceleb1-indian-demo"
    assert info["expected_recording_count"] == dataset.EXPECTED_RECORDING_COUNT
    assert info["total_recordings"] == len(FAKE_FILES)
    assert info["available"] is True
    assert ".wav" in info["audio_extensions"]
    _assert_no_ground_truth_leak(info)


def test_list_recordings_excludes_non_audio_files(fake_dataset_dir):
    recordings = dataset.list_recordings()
    assert len(recordings) == len(FAKE_FILES)
    assert all(r.extension == ".wav" for r in recordings)


def test_duration_seconds_is_none_for_unreadable_audio(fake_dataset_dir):
    # FAKE_FILES are non-audio bytes, so torchaudio.info() cannot read them —
    # this exercises the except-Exception -> None fallback, not the happy path.
    recordings = dataset.list_recordings()
    assert len(recordings) > 0
    for recording in recordings:
        assert recording.duration_seconds is None
        assert "duration_seconds" in asdict(recording)


def test_duration_seconds_is_positive_for_real_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    dataset_dir = tmp_path / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)

    sample_rate = 16000
    seconds = 0.5
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(dataset_dir / "id10018_01.wav", audio, sample_rate)

    recordings = dataset.list_recordings()
    assert len(recordings) == 1
    assert recordings[0].duration_seconds == pytest.approx(seconds, abs=0.05)


def test_recording_ids_are_stable_and_opaque(fake_dataset_dir):
    first_pass = {r.recording_id: r for r in dataset.list_recordings()}
    second_pass = {r.recording_id: r for r in dataset.list_recordings()}

    assert set(first_pass) == set(second_pass)
    for recording_id, recording in first_pass.items():
        assert recording_id.startswith("rec_")
        assert recording.display_filename == f"{recording_id}{recording.extension}"

    _assert_no_ground_truth_leak([asdict(r) for r in first_pass.values()])


def test_same_speaker_recordings_get_distinct_ids(fake_dataset_dir):
    recordings = dataset.list_recordings()
    ids = [r.recording_id for r in recordings]
    assert len(ids) == len(set(ids))


def test_get_recording_returns_matching_entry(fake_dataset_dir):
    known = dataset.list_recordings()[0]
    fetched = dataset.get_recording(known.recording_id)
    assert fetched == known


@pytest.mark.parametrize(
    "bad_id",
    ["not-a-real-id", "../../etc/passwd", "id10018_01.wav", "", "rec_0000000000000000"],
)
def test_get_recording_rejects_unknown_and_traversal_ids(fake_dataset_dir, bad_id):
    with pytest.raises(dataset.RecordingNotFound):
        dataset.get_recording(bad_id)


def test_speaker_prefix_for_extracts_id_prefix():
    assert dataset._speaker_prefix_for("id10018_01.wav") == "id10018"
    assert dataset._speaker_prefix_for("id123_x.wav") == "id123"


def test_speaker_prefix_for_returns_none_for_non_matching_filename():
    assert dataset._speaker_prefix_for("upload-000.wav") is None
    assert dataset._speaker_prefix_for("randomfile.mp3") is None
    assert dataset._speaker_prefix_for("") is None


def test_recordings_sharing_prefix_share_speaker_group_id(fake_dataset_dir):
    recordings_by_id = {r.recording_id: r for r in dataset.list_recordings()}
    group_for = lambda name: recordings_by_id[dataset._recording_id_for(name)].speaker_group_id

    # FAKE_FILES: id10018_01.wav and id10018_02.wav share prefix id10018;
    # id10324_01.wav has a different prefix.
    assert group_for("id10018_01.wav") == group_for("id10018_02.wav")
    assert group_for("id10018_01.wav") != group_for("id10324_01.wav")
    assert group_for("id10018_01.wav") is not None
    assert group_for("id10324_01.wav") is not None


def test_speaker_group_assignment_is_deterministic(fake_dataset_dir):
    first = {r.recording_id: r.speaker_group_id for r in dataset.list_recordings()}
    second = {r.recording_id: r.speaker_group_id for r in dataset.list_recordings()}
    assert first == second


def test_discover_group_order_follows_recording_id_sort_not_filename_order(monkeypatch, tmp_path):
    # Filenames chosen so alphabetical filename order (id100_a, id100_b,
    # id200_a, id200_b) disagrees with opaque-recording-id sort order
    # (id200_b, id100_a, id100_b, id200_a) -- verified by direct sha256
    # computation. This proves group numbering follows the sorted-id order
    # dataset._discover already uses, not filename order, so no filename
    # ordering leaks into the anonymous group ids.
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    filenames = ["id100_a.wav", "id100_b.wav", "id200_a.wav", "id200_b.wav"]
    _build_dataset_with_filenames(tmp_path, filenames)

    recording_ids_sorted = [r.recording_id for r in dataset.list_recordings()]
    expected_sorted_names = ["id200_b.wav", "id100_a.wav", "id100_b.wav", "id200_a.wav"]
    assert recording_ids_sorted == [dataset._recording_id_for(name) for name in expected_sorted_names]

    recordings_by_id = {r.recording_id: r for r in dataset.list_recordings()}
    group_for = lambda name: recordings_by_id[dataset._recording_id_for(name)].speaker_group_id

    # id200 appears first in recording-id-sorted order (via id200_b) -> group 1.
    # id100 appears second (via id100_a) -> group 2. Alphabetical filename
    # order would have put id100 first -- this proves it doesn't.
    assert group_for("id200_b.wav") == "speaker-group-1"
    assert group_for("id200_a.wav") == "speaker-group-1"
    assert group_for("id100_a.wav") == "speaker-group-2"
    assert group_for("id100_b.wav") == "speaker-group-2"


def test_discover_speaker_group_id_is_none_for_unparseable_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    _build_dataset_with_filenames(tmp_path, ["id10018_01.wav", "upload-000.wav"])

    recordings_by_id = {r.recording_id: r for r in dataset.list_recordings()}
    valid_id = dataset._recording_id_for("id10018_01.wav")
    stray_id = dataset._recording_id_for("upload-000.wav")

    assert recordings_by_id[valid_id].speaker_group_id is not None
    assert recordings_by_id[stray_id].speaker_group_id is None


def test_get_speaker_group_map_omits_recordings_without_group(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    _build_dataset_with_filenames(tmp_path, ["id10018_01.wav", "upload-000.wav"])

    mapping = dataset.get_speaker_group_map()
    valid_id = dataset._recording_id_for("id10018_01.wav")
    stray_id = dataset._recording_id_for("upload-000.wav")

    assert mapping.get(valid_id) is not None
    assert mapping.get(stray_id) is None
    assert mapping.get("rec_totally_unknown_id") is None


def test_get_speaker_group_map_returns_empty_when_dataset_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    assert dataset.get_speaker_group_map() == {}


def test_missing_dataset_directory_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)

    info = dataset.get_dataset_info()
    assert info["available"] is False
    assert info["total_recordings"] == 0

    with pytest.raises(dataset.DatasetUnavailable):
        dataset.list_recordings()

    with pytest.raises(dataset.DatasetUnavailable):
        dataset.get_recording("rec_anything")


@pytest.mark.asyncio
async def test_dataset_info_endpoint(client, fake_dataset_dir):
    response = await client.get("/tasks/verification/dataset")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "voxceleb1-indian-demo"
    assert body["total_recordings"] == len(FAKE_FILES)
    _assert_no_ground_truth_leak(body)


@pytest.mark.asyncio
async def test_dataset_recordings_endpoint(client, fake_dataset_dir):
    response = await client.get("/tasks/verification/dataset/recordings")
    assert response.status_code == 200
    body = response.json()
    assert body["total_recordings"] == len(FAKE_FILES)
    assert len(body["recordings"]) == len(FAKE_FILES)
    _assert_no_ground_truth_leak(body)


@pytest.mark.asyncio
async def test_dataset_single_recording_endpoint(client, fake_dataset_dir):
    listing = await client.get("/tasks/verification/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]

    response = await client.get(f"/tasks/verification/dataset/recordings/{recording_id}")
    assert response.status_code == 200
    assert response.json()["recording_id"] == recording_id


@pytest.mark.asyncio
async def test_dataset_single_recording_endpoint_rejects_unknown_id(client, fake_dataset_dir):
    response = await client.get("/tasks/verification/dataset/recordings/not-a-real-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_dataset_recordings_endpoint_never_exposes_speaker_group_id(client, fake_dataset_dir):
    response = await client.get("/tasks/verification/dataset/recordings")
    assert response.status_code == 200
    body = response.json()
    assert len(body["recordings"]) == len(FAKE_FILES)
    for recording in body["recordings"]:
        assert "speaker_group_id" not in recording


@pytest.mark.asyncio
async def test_dataset_single_recording_endpoint_never_exposes_speaker_group_id(client, fake_dataset_dir):
    listing = await client.get("/tasks/verification/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]

    response = await client.get(f"/tasks/verification/dataset/recordings/{recording_id}")
    assert response.status_code == 200
    assert "speaker_group_id" not in response.json()


@pytest.mark.asyncio
async def test_dataset_endpoints_404_when_dataset_missing(monkeypatch, client, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)

    recordings_response = await client.get("/tasks/verification/dataset/recordings")
    assert recordings_response.status_code == 404

    single_response = await client.get("/tasks/verification/dataset/recordings/rec_anything")
    assert single_response.status_code == 404


@pytest.mark.skipif(
    not REAL_DATASET_DIR.is_dir(),
    reason="Real gitignored demo dataset is not present in this environment",
)
def test_real_dataset_has_exactly_92_recordings():
    recordings = dataset.list_recordings()
    assert len(recordings) == dataset.EXPECTED_RECORDING_COUNT


def _assert_no_ground_truth_leak_headers(headers) -> None:
    _assert_no_ground_truth_leak(dict(headers))


@pytest.mark.asyncio
async def test_dataset_recording_audio_returns_exact_bytes_and_headers(client, fake_dataset_dir):
    listing = await client.get("/tasks/verification/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]

    response = await client.get(f"/tasks/verification/dataset/recordings/{recording_id}/audio")

    assert response.status_code == 200
    assert response.content == b"RIFF-fake-audio"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == f'inline; filename="{recording_id}.wav"'
    _assert_no_ground_truth_leak_headers(response.headers)


@pytest.mark.asyncio
async def test_dataset_recording_audio_supports_range_requests(client, fake_dataset_dir):
    listing = await client.get("/tasks/verification/dataset/recordings")
    recording_id = listing.json()["recordings"][0]["recording_id"]

    response = await client.get(
        f"/tasks/verification/dataset/recordings/{recording_id}/audio",
        headers={"Range": "bytes=0-3"},
    )

    assert response.status_code == 206
    assert response.content == b"RIFF-fake-audio"[:4]
    assert response.headers["content-range"] == "bytes 0-3/15"
    assert response.headers["content-length"] == "4"
    _assert_no_ground_truth_leak_headers(response.headers)


@pytest.mark.parametrize(
    "bad_id",
    ["not-a-real-id", "../../etc/passwd", "id10018_01.wav", "", "rec_0000000000000000"],
)
@pytest.mark.asyncio
async def test_dataset_recording_audio_rejects_unknown_and_traversal_ids(client, fake_dataset_dir, bad_id):
    response = await client.get(f"/tasks/verification/dataset/recordings/{bad_id}/audio")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_dataset_recording_audio_404_when_dataset_missing(monkeypatch, client, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)

    response = await client.get("/tasks/verification/dataset/recordings/rec_anything/audio")
    assert response.status_code == 404
