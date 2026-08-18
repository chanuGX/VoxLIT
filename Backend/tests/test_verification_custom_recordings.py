"""Focused tests for Speaker Verification's `crec_...` custom-recording
identity layer: deterministic/stable ids, session isolation, multi-dataset
Redis-mapping isolation (no clobbering across datasets/tabs), Redis-failure
handling (503, distinct from a clean 404 miss), live revalidation on
deletion, collision detection, Range/206 playback, and end-to-end flow
through `_resolve_audio_source` into batch analysis, pair verification,
enrollment/cluster saliency, and perturbation.

All custom-dataset storage is isolated to `tmp_path` (mirrors
`test_custom_dataset_service.py`); Redis is the `fakeredis` instance the
global `fake_redis` autouse fixture (see conftest.py) already installs.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient
from redis.exceptions import RedisError

from app.core import redis as redis_module
from app.core.settings import settings
from app.main import app
from app.services import custom_dataset_service as cds
from app.tasks.verification import custom_recordings, dataset


def _wav_bytes(*, sample_rate: int = 16000, seconds: float = 0.5, freq: float = 220.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch, tmp_path):
    """Every test gets its own custom-dataset storage root -- mirrors
    test_custom_dataset_service.py's isolated_storage fixture."""
    root = tmp_path / "sessions"
    monkeypatch.setattr(cds, "SESSIONS_BASE_DIR", root)
    return root


FAKE_DEMO_FILES = [f"speaker{i}_clip.wav" for i in range(4)]


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    """A real (small, synthetic) demo dataset, for tests mixing `rec_` ids
    with `crec_` ids in one request."""
    demo_root = tmp_path / "demo"
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", demo_root)
    dataset_dir = demo_root / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    content = _wav_bytes()
    for filename in FAKE_DEMO_FILES:
        (dataset_dir / filename).write_bytes(content)
    return [recording.recording_id for recording in dataset.list_recordings()]


async def _create_custom_dataset(client, dataset_name: str = "My Voices") -> str:
    """Creates a dataset via the real /upload/dataset/create route and
    returns its bare name -- `original_name`, never `dataset_name` (which is
    the formatted, session-embedding string for this specific endpoint)."""

    response = await client.post("/upload/dataset/create", data={"dataset_name": dataset_name})
    assert response.status_code == 201
    return response.json()["original_name"]


async def _upload_file(client, dataset_name: str, filename: str = "clip.wav", *, freq: float = 220.0) -> None:
    response = await client.post(
        f"/upload/dataset/{dataset_name}/files",
        files={"files": (filename, io.BytesIO(_wav_bytes(freq=freq)), "audio/wav")},
    )
    assert response.status_code == 200


async def _create_dataset_with_recordings(
    client, dataset_name: str = "My Voices", filenames: tuple[str, ...] = ("clip.wav",)
) -> list[dict]:
    await _create_custom_dataset(client, dataset_name)
    for index, filename in enumerate(filenames):
        await _upload_file(client, dataset_name, filename, freq=220.0 + index * 40.0)
    listing = await client.get(f"/tasks/verification/dataset/custom/{dataset_name}/recordings")
    assert listing.status_code == 200
    return listing.json()["recordings"]


def _assert_no_leak(payload: object, *, forbidden_substrings: list[str]) -> None:
    serialized = json.dumps(payload)
    for value in forbidden_substrings:
        assert value not in serialized


# ---------------------------------------------------------------------------
# Identity: deterministic, stable, safe fields only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crec_id_deterministic_and_stable_across_relistings(client):
    await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    first = await client.get("/tasks/verification/dataset/custom/My Voices/recordings")
    second = await client.get("/tasks/verification/dataset/custom/My Voices/recordings")

    first_id = first.json()["recordings"][0]["recording_id"]
    second_id = second.json()["recordings"][0]["recording_id"]
    assert first_id == second_id
    assert first_id.startswith("crec_")


@pytest.mark.asyncio
async def test_custom_recording_listing_has_safe_fields_only(client):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    entry = recordings[0]
    assert set(entry.keys()) == {
        "recording_id",
        "display_filename",
        "extension",
        "size_bytes",
        "duration_seconds",
        "dataset_name",
        "origin",
    }
    assert entry["origin"] == "custom_dataset"
    assert entry["dataset_name"] == "My Voices"
    assert entry["extension"] == ".wav"
    assert entry["size_bytes"] > 0
    _assert_no_leak(entry, forbidden_substrings=["custom:", "sessions"])


@pytest.mark.asyncio
async def test_safe_dataset_listing_contains_only_safe_fields(client):
    await _create_custom_dataset(client, "My Voices")
    response = await client.get("/tasks/verification/dataset/custom")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert len(datasets) == 1
    assert set(datasets[0].keys()) == {"dataset_name", "total_files", "created_at"}
    assert datasets[0]["dataset_name"] == "My Voices"
    assert datasets[0]["total_files"] == 0


@pytest.mark.asyncio
async def test_safe_dataset_listing_never_leaks_session_id(client):
    # The session cookie is only set on the response to the request that
    # first establishes it -- capture it here, before any later request
    # (which will already be carrying the cookie) stops re-issuing it.
    models_response = await client.get("/tasks/verification/models")
    sid = models_response.cookies.get(settings.SESSION_COOKIE_NAME)
    assert sid is not None

    await _create_custom_dataset(client, "My Voices")

    response = await client.get("/tasks/verification/dataset/custom")
    _assert_no_leak(response.json(), forbidden_substrings=[sid, "custom:", "formatted_name", "session_id"])


# ---------------------------------------------------------------------------
# Owner can list, resolve, play; playback supports Range/206
# ---------------------------------------------------------------------------


async def _assert_streams_correctly(client, url: str, expected_bytes: bytes) -> None:
    response = await client.get(url)
    assert response.status_code == 200
    assert response.content == expected_bytes
    assert response.headers.get("accept-ranges") == "bytes"

    ranged = await client.get(url, headers={"Range": "bytes=0-3"})
    assert ranged.status_code == 206
    assert ranged.content == expected_bytes[:4]
    assert ranged.headers.get("content-range") == f"bytes 0-3/{len(expected_bytes)}"

    head = await client.head(url)
    assert head.status_code == 200
    assert head.content == b""


@pytest.mark.asyncio
async def test_owner_can_resolve_and_play_custom_recording_with_range_support(client):
    content = _wav_bytes(freq=333.0)
    await _create_custom_dataset(client, "My Voices")
    response = await client.post(
        "/upload/dataset/My Voices/files",
        files={"files": ("clip.wav", io.BytesIO(content), "audio/wav")},
    )
    assert response.status_code == 200

    recordings = (await client.get("/tasks/verification/dataset/custom/My Voices/recordings")).json()["recordings"]
    recording_id = recordings[0]["recording_id"]

    await _assert_streams_correctly(
        client, f"/tasks/verification/custom-recordings/{recording_id}/audio", content
    )


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_session_cannot_list_resolve_or_play_owner_dataset(client):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    recording_id = recordings[0]["recording_id"]

    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as other_client:
        listing = await other_client.get("/tasks/verification/dataset/custom/My Voices/recordings")
        assert listing.status_code == 404

        audio = await other_client.get(f"/tasks/verification/custom-recordings/{recording_id}/audio")
        assert audio.status_code == 404

        safe_list = await other_client.get("/tasks/verification/dataset/custom")
        assert safe_list.status_code == 200
        assert safe_list.json()["datasets"] == []


# ---------------------------------------------------------------------------
# Malformed / unknown ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_crec_id_is_404(client):
    response = await client.get(
        f"/tasks/verification/custom-recordings/crec_{'0' * 64}/audio"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_malformed_crec_id_is_404(client):
    response = await client.get("/tasks/verification/custom-recordings/crec_not-hex/audio")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unknown_custom_dataset_name_is_404(client):
    response = await client.get("/tasks/verification/dataset/custom/does-not-exist/recordings")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Multi-dataset / multi-tab isolation -- the corrected per-recording-key design
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listing_second_dataset_does_not_invalidate_first(client):
    recordings_a = await _create_dataset_with_recordings(client, "Dataset A", ("a.wav",))
    id_a = recordings_a[0]["recording_id"]

    # Listing (and thus re-writing Redis mappings for) a second dataset must
    # never touch the first dataset's own mapping keys.
    await _create_dataset_with_recordings(client, "Dataset B", ("b.wav",))

    audio_a = await client.get(f"/tasks/verification/custom-recordings/{id_a}/audio")
    assert audio_a.status_code == 200


# ---------------------------------------------------------------------------
# Deletion invalidates recordings via live revalidation, even within TTL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deleting_dataset_invalidates_its_recordings_immediately(client):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    recording_id = recordings[0]["recording_id"]

    # The Redis mapping is still well within its TTL -- deletion must still
    # 404 via the live filesystem recheck in resolve_custom_recording_path.
    delete_response = await client.delete("/upload/dataset/My Voices")
    assert delete_response.status_code == 200

    audio = await client.get(f"/tasks/verification/custom-recordings/{recording_id}/audio")
    assert audio.status_code == 404

    listing = await client.get("/tasks/verification/dataset/custom/My Voices/recordings")
    assert listing.status_code == 404


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forced_id_collision_fails_closed(client, monkeypatch):
    monkeypatch.setattr(custom_recordings, "_recording_id_for", lambda sid, name, filename: "crec_" + "a" * 64)
    await _create_custom_dataset(client, "My Voices")
    await _upload_file(client, "My Voices", "a.wav", freq=220.0)
    await _upload_file(client, "My Voices", "b.wav", freq=440.0)

    response = await client.get("/tasks/verification/dataset/custom/My Voices/recordings")
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Redis-unavailable (503) vs missing-mapping (404), distinguished, including
# the sliding-TTL EXPIRE refresh step
# ---------------------------------------------------------------------------


class _FailingPipeline:
    def set(self, *args, **kwargs):
        return self

    async def execute(self):
        raise RedisError("simulated redis outage")


async def _raising(*args, **kwargs):
    raise RedisError("simulated redis outage")


class _RedisFailureWrapper:
    """Delegates every attribute to the real, shared redis client except the
    ones explicitly overridden -- isolates a simulated Redis failure to
    exactly the operation under test. Patched in as `custom_recordings`'s
    own `redis_module` name (not the real `app.core.redis` module), so
    SessionMiddleware's own unrelated pipeline usage on the same shared
    client is completely unaffected."""

    def __init__(self, real, **overrides):
        self._real = real
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._real, name)


def _install_custom_recordings_redis_failure(monkeypatch, **overrides) -> None:
    stub_module = type("StubRedisModule", (), {})()
    stub_module.redis = _RedisFailureWrapper(redis_module.redis, **overrides)
    monkeypatch.setattr(custom_recordings, "redis_module", stub_module)


@pytest.mark.asyncio
async def test_listing_returns_503_when_redis_pipeline_write_fails(client, monkeypatch):
    await _create_custom_dataset(client, "My Voices")
    await _upload_file(client, "My Voices", "clip.wav")

    _install_custom_recordings_redis_failure(monkeypatch, pipeline=lambda *a, **k: _FailingPipeline())
    response = await client.get("/tasks/verification/dataset/custom/My Voices/recordings")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_resolve_returns_503_when_redis_get_fails(client, monkeypatch):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    recording_id = recordings[0]["recording_id"]

    _install_custom_recordings_redis_failure(monkeypatch, get=_raising)
    response = await client.get(f"/tasks/verification/custom-recordings/{recording_id}/audio")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_resolve_returns_503_when_ttl_refresh_fails_after_successful_resolve(client, monkeypatch):
    """A successful mapping lookup and a successful filesystem resolve must
    still surface as 503 -- not a silently-degraded 200 with a stale TTL,
    and not an unhandled 500 -- if the trailing EXPIRE call fails."""

    recordings = await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    recording_id = recordings[0]["recording_id"]

    real_get = redis_module.redis.get
    _install_custom_recordings_redis_failure(monkeypatch, get=real_get, expire=_raising)
    response = await client.get(f"/tasks/verification/custom-recordings/{recording_id}/audio")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_resolve_clean_miss_is_404_not_503(client):
    """With Redis fully healthy, an id that was simply never listed is a
    plain 404 -- the failure-mode tests above must not be conflating this
    with an infra problem."""

    response = await client.get(
        f"/tasks/verification/custom-recordings/crec_{'1' * 64}/audio"
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# End-to-end: crec_ flows through _resolve_audio_source into every
# id-accepting endpoint, mixed with rec_/asset_ where applicable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_dataset_accepts_crec_mixed_with_demo_and_session_asset_ids(client, fake_dataset_dir):
    demo_id = fake_dataset_dir[0]
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["asset_id"]

    recordings = await _create_dataset_with_recordings(client, "My Voices", ("custom.wav",))
    crec_id = recordings[0]["recording_id"]

    mixed_ids = [demo_id, asset_id, crec_id]
    expected = {"model": "ecapa-tdnn", "labels": mixed_ids, "recording_count": 3}
    with patch("app.tasks.verification.router._cached_batch_analysis", return_value=expected) as mock_analysis:
        response = await client.post(
            "/tasks/verification/batch/dataset",
            json={"model": "ecapa-tdnn", "recording_ids": mixed_ids},
        )

    assert response.status_code == 200
    assert response.json() == expected
    assert mock_analysis.call_args.args[2] == mixed_ids


@pytest.mark.asyncio
async def test_batch_dataset_rejects_crec_from_deleted_dataset(client):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("a.wav", "b.wav"))
    ids = [r["recording_id"] for r in recordings]
    await client.delete("/upload/dataset/My Voices")

    response = await client.post(
        "/tasks/verification/batch/dataset",
        json={"model": "ecapa-tdnn", "recording_ids": ids},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_verify_endpoint_accepts_crec_enrollment_and_probe(client):
    recordings = await _create_dataset_with_recordings(
        client, "My Voices", ("a.wav", "b.wav", "c.wav", "probe.wav")
    )
    ids = [r["recording_id"] for r in recordings]
    enrollment_ids, probe_id = ids[:3], ids[3]

    expected = {"model": "ecapa-tdnn", "similarity": 0.9, "threshold": 0.35, "same_speaker": True}
    with patch("app.tasks.verification.router.verify_speaker", return_value=expected):
        response = await client.post(
            "/tasks/verification/verify",
            data={
                "model": "ecapa-tdnn",
                "enrollment_recording_ids": enrollment_ids,
                "probe_recording_id": probe_id,
            },
        )

    assert response.status_code == 200
    assert response.json() == expected


_SALIENCY_RESULT = {
    "model": "ecapa-tdnn",
    "model_label": "ECAPA-TDNN",
    "reference_type": "cluster",
    "cluster_id": "cluster-1",
    "target_recording_id": None,
    "reference_count": 2,
    "baseline_similarity": 0.9,
    "threshold": 0.35,
    "segment_count": 8,
    "audio_duration_seconds": 0.5,
    "interpretation": "irrelevant for this test",
    "segments": [],
}


@pytest.mark.asyncio
async def test_saliency_enrollment_mode_accepts_crec_enrollment_and_probe(client):
    recordings = await _create_dataset_with_recordings(
        client, "My Voices", ("a.wav", "b.wav", "c.wav", "probe.wav")
    )
    ids = [r["recording_id"] for r in recordings]
    expected = {**_SALIENCY_RESULT, "reference_type": "enrollment", "cluster_id": None, "target_recording_id": ids[3]}

    with patch("app.tasks.verification.router.compute_saliency_map", return_value=expected):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={
                "model": "ecapa-tdnn",
                "reference_type": "enrollment",
                "enrollment_recording_ids": ids[:3],
                "probe_recording_id": ids[3],
            },
        )

    assert response.status_code == 200
    assert response.json()["reference_type"] == "enrollment"


@pytest.mark.asyncio
async def test_saliency_cluster_mode_accepts_crec_references_and_target(client):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("a.wav", "b.wav", "target.wav"))
    ids = [r["recording_id"] for r in recordings]
    reference_ids, target_id = ids[:2], ids[2]

    with patch("app.tasks.verification.router.compute_saliency_map", return_value=_SALIENCY_RESULT):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={
                "model": "ecapa-tdnn",
                "reference_type": "cluster",
                "reference_recording_ids": reference_ids,
                "target_recording_id": target_id,
                "cluster_id": "cluster-1",
            },
        )

    assert response.status_code == 200
    assert response.json()["reference_type"] == "cluster"


def _write_wav(path, *, sample_rate: int = 16000, seconds: float = 0.5, freq: float = 250.0):
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sample_rate)
    return path


def _fake_router_perturbation_result(**overrides) -> dict:
    result = {
        "model": "ecapa-tdnn",
        "model_label": "ECAPA-TDNN",
        "threshold": 0.35,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05, "seed": 0}},
        "similarity": 0.8,
        "same_speaker": True,
        "embedding_shift": 0.2,
        "original_embedding": [0.1] * 192,
        "perturbed_embedding": [0.1] * 192,
    }
    result.update(overrides)
    return result


def _fake_perturb_and_compare(model_key, source_path, perturbation_type, perturbation_params, output_path, cached_original_embedding=None):
    _write_wav(output_path, freq=444.0)
    return _fake_router_perturbation_result()


@pytest.mark.asyncio
async def test_perturbation_accepts_crec_source_and_leaves_original_untouched(client):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    recording_id = recordings[0]["recording_id"]

    payload = {
        "model": "ecapa-tdnn",
        "recording_id": recording_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    with patch("app.tasks.verification.router.perturb_and_compare", side_effect=_fake_perturb_and_compare):
        response = await client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["source_recording_id"] == recording_id
    assert body["session_asset"]["asset_id"].startswith("asset_")
    assert body["session_asset"]["origin"] == "perturbation"
    assert "original_embedding" not in body
    assert "perturbed_embedding" not in body

    # The original custom-dataset recording is untouched -- still resolvable
    # under its own crec_ id, distinct from the newly created asset_ output.
    still_there = await client.get(f"/tasks/verification/custom-recordings/{recording_id}/audio")
    assert still_there.status_code == 200


@pytest.mark.asyncio
async def test_perturbation_rejects_crec_from_another_session(client):
    recordings = await _create_dataset_with_recordings(client, "My Voices", ("clip.wav",))
    recording_id = recordings[0]["recording_id"]

    payload = {
        "model": "ecapa-tdnn",
        "recording_id": recording_id,
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as other_client:
        response = await other_client.post("/tasks/verification/perturbation", json=payload)

    assert response.status_code == 404
