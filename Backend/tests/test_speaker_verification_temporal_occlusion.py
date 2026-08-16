"""Focused tests for the Speaker Verification temporal-occlusion saliency endpoint."""

import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient

from app.core.settings import settings
from app.main import app
from app.tasks.verification import dataset


def _wav_bytes(*, sample_rate: int = 16000, seconds: float = 1.0, freq: float = 220.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _audio_upload(name: str):
    return (name, io.BytesIO(_wav_bytes()), "audio/wav")


FAKE_FILES = [f"speaker{i}_clip.wav" for i in range(4)]


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    dataset_dir = tmp_path / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    for index, filename in enumerate(FAKE_FILES):
        (dataset_dir / filename).write_bytes(b"RIFF-fake-audio")
    return [recording.recording_id for recording in dataset.list_recordings()]


_EXPECTED_RESULT = {
    "model": "ecapa-tdnn",
    "baseline_similarity": 0.9,
    "threshold": 0.3578562438488006,
    "segment_count": 8,
    "segments": [{"segment_index": i + 1, "importance": 0.01 * i} for i in range(8)],
}


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_requires_three_enrollment_files(client):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    response = await client.post(
        "/tasks/verification/explain/temporal-occlusion",
        data={"model": "ecapa-tdnn"},
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_rejects_out_of_range_segment_count(client):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("enrollment_files", _audio_upload("three.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    response = await client.post(
        "/tasks/verification/explain/temporal-occlusion",
        data={"model": "ecapa-tdnn", "segment_count": "2"},
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_rejects_invalid_model(client):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("enrollment_files", _audio_upload("three.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    response = await client.post(
        "/tasks/verification/explain/temporal-occlusion",
        data={"model": "not-a-real-model"},
        files=files,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_upload_happy_path(client):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("enrollment_files", _audio_upload("three.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    with patch(
        "app.tasks.verification.router.temporal_occlusion_saliency", return_value=_EXPECTED_RESULT
    ):
        response = await client.post(
            "/tasks/verification/explain/temporal-occlusion",
            data={"model": "ecapa-tdnn"},
            files=files,
        )

    assert response.status_code == 200
    assert response.json() == _EXPECTED_RESULT


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_accepts_mixed_demo_and_session_asset_ids(client, fake_dataset_dir):
    enrollment_ids = fake_dataset_dir[:3]
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert upload.status_code == 200
    probe_id = upload.json()["asset_id"]

    with patch(
        "app.tasks.verification.router.temporal_occlusion_saliency", return_value=_EXPECTED_RESULT
    ) as mock_saliency:
        response = await client.post(
            "/tasks/verification/explain/temporal-occlusion",
            data={
                "model": "ecapa-tdnn",
                "enrollment_recording_ids": enrollment_ids,
                "probe_recording_id": probe_id,
            },
        )

    assert response.status_code == 200
    assert response.json() == _EXPECTED_RESULT
    call_args = mock_saliency.call_args.args
    assert call_args[0] == "ecapa-tdnn"
    assert len(call_args[1]) == 3
    assert str(call_args[2]).endswith(".wav")


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_rejects_ids_and_files_together(client, fake_dataset_dir):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("enrollment_files", _audio_upload("three.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    response = await client.post(
        "/tasks/verification/explain/temporal-occlusion",
        data={
            "model": "ecapa-tdnn",
            "enrollment_recording_ids": fake_dataset_dir[:3],
            "probe_recording_id": fake_dataset_dir[3],
        },
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_requires_ids_or_files(client):
    response = await client.post(
        "/tasks/verification/explain/temporal-occlusion",
        data={"model": "ecapa-tdnn"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_temporal_occlusion_endpoint_rejects_cross_session_asset_id(client, fake_dataset_dir):
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    probe_id = upload.json()["asset_id"]

    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as other_client:
        response = await other_client.post(
            "/tasks/verification/explain/temporal-occlusion",
            data={
                "model": "ecapa-tdnn",
                "enrollment_recording_ids": fake_dataset_dir[:3],
                "probe_recording_id": probe_id,
            },
        )

    assert response.status_code == 404
