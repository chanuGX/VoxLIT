"""Focused tests for the generalized Speaker Verification saliency-map
endpoint (`/tasks/verification/explain/saliency`) and its underlying
`service.compute_saliency_map`/`service._occlude_and_score` core.

Endpoint-level tests mock the service layer and assert request wiring and
validation, mirroring `test_speaker_verification_temporal_occlusion.py`.
Math-level tests call `service.compute_saliency_map` directly against real
(synthetic) WAV files with a content-dependent fake adapter, mirroring
`test_speaker_verification_batch.py`'s `_FakeAdapter` pattern.
"""

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from httpx import AsyncClient

from app.core.settings import settings
from app.main import app
from app.tasks.verification import clustering, dataset, service


def _wav_bytes(*, sample_rate: int = 16000, seconds: float = 1.0, freq: float = 220.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _audio_upload(name: str):
    return (name, io.BytesIO(_wav_bytes()), "audio/wav")


def _quarter_amplitude_wav_bytes(
    amplitudes: tuple[float, float, float, float],
    *,
    sample_rate: int = 16000,
    seconds: float = 1.0,
) -> bytes:
    """A WAV whose four quarters each hold a constant absolute amplitude
    (not a sine wave, so the mean-abs-amplitude of each quarter is exactly
    the requested value, with no phase ambiguity) -- for deterministic
    content-dependent fake-embedding tests."""

    total_samples = int(sample_rate * seconds)
    quarter = total_samples // 4
    chunks = []
    for index, amplitude in enumerate(amplitudes):
        length = total_samples - quarter * 3 if index == 3 else quarter
        chunks.append(np.full(length, amplitude, dtype=np.float32))
    audio = np.concatenate(chunks)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _empty_wav_bytes(*, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.array([], dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


FAKE_FILES = [f"speaker{i}_clip.wav" for i in range(4)]


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    dataset_dir = tmp_path / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    for filename in FAKE_FILES:
        (dataset_dir / filename).write_bytes(b"RIFF-fake-audio")
    return [recording.recording_id for recording in dataset.list_recordings()]


_EXPECTED_RESULT = {
    "model": "ecapa-tdnn",
    "model_label": "ECAPA-TDNN",
    "reference_type": "cluster",
    "cluster_id": "cluster-1",
    "target_recording_id": "rec_target",
    "reference_count": 2,
    "baseline_similarity": 0.9,
    "threshold": 0.3578562438488006,
    "segment_count": 8,
    "audio_duration_seconds": 1.0,
    "interpretation": "irrelevant for this test",
    "segments": [
        {
            "segment_index": i + 1,
            "start_seconds": i * 0.125,
            "end_seconds": (i + 1) * 0.125,
            "occluded_similarity": 0.9 - 0.01 * i,
            "similarity_change": 0.01 * i,
            "influence_strength": abs(0.01 * i),
        }
        for i in range(8)
    ],
}


async def _fake_resolve_audio_source(recording_id: str, sid: str) -> Path:
    return Path("/fake") / f"{recording_id}.wav"


# ---------------------------------------------------------------------------
# Endpoint-level: basic validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saliency_endpoint_rejects_invalid_reference_type(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={"model": "ecapa-tdnn", "reference_type": "not-a-mode"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_rejects_out_of_range_segment_count(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={"model": "ecapa-tdnn", "reference_type": "cluster", "segment_count": "2"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_rejects_invalid_model(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "not-a-real-model",
            "reference_type": "cluster",
            "reference_recording_ids": ["rec_a"],
            "target_recording_id": "rec_target",
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Endpoint-level: cluster mode reference-count/duplicate/overlap bounds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_accepts_single_reference_id(client):
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", return_value=_EXPECTED_RESULT),
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={
                "model": "ecapa-tdnn",
                "reference_type": "cluster",
                "reference_recording_ids": ["rec_a"],
                "target_recording_id": "rec_target",
                "cluster_id": "cluster-1",
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_accepts_max_reference_ids(client):
    reference_ids = [f"rec_{i:03d}" for i in range(99)]
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", return_value=_EXPECTED_RESULT),
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={
                "model": "ecapa-tdnn",
                "reference_type": "cluster",
                "reference_recording_ids": reference_ids,
                "target_recording_id": "rec_target",
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_over_max_reference_ids(client):
    reference_ids = [f"rec_{i:03d}" for i in range(100)]
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "cluster",
            "reference_recording_ids": reference_ids,
            "target_recording_id": "rec_target",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_duplicate_reference_ids(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "cluster",
            "reference_recording_ids": ["rec_a", "rec_b", "rec_a"],
            "target_recording_id": "rec_target",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_empty_reference_ids(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "cluster",
            "target_recording_id": "rec_target",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_missing_target_id(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "cluster",
            "reference_recording_ids": ["rec_a", "rec_b"],
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_target_also_in_references(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "cluster",
            "reference_recording_ids": ["rec_a", "rec_target"],
            "target_recording_id": "rec_target",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_cross_session_asset_id(client, fake_dataset_dir):
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert upload.status_code == 200
    foreign_asset_id = upload.json()["asset_id"]

    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as other_client:
        response = await other_client.post(
            "/tasks/verification/explain/saliency",
            data={
                "model": "ecapa-tdnn",
                "reference_type": "cluster",
                "reference_recording_ids": [foreign_asset_id],
                "target_recording_id": fake_dataset_dir[0],
            },
        )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Endpoint-level: strict mutual exclusivity between reference_type fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_enrollment_ids_present(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "cluster",
            "reference_recording_ids": ["rec_a", "rec_b"],
            "target_recording_id": "rec_target",
            "enrollment_recording_ids": ["rec_c", "rec_d", "rec_e"],
            "probe_recording_id": "rec_probe",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_cluster_mode_rejects_uploaded_files(client):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "cluster",
            "reference_recording_ids": ["rec_a", "rec_b"],
            "target_recording_id": "rec_target",
        },
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_enrollment_mode_rejects_reference_recording_ids_present(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "enrollment",
            "enrollment_recording_ids": ["rec_a", "rec_b", "rec_c"],
            "probe_recording_id": "rec_probe",
            "reference_recording_ids": ["rec_x"],
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_enrollment_mode_rejects_target_recording_id_present(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "enrollment",
            "enrollment_recording_ids": ["rec_a", "rec_b", "rec_c"],
            "probe_recording_id": "rec_probe",
            "target_recording_id": "rec_x",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_enrollment_mode_rejects_cluster_id_present(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "enrollment",
            "enrollment_recording_ids": ["rec_a", "rec_b", "rec_c"],
            "probe_recording_id": "rec_probe",
            "cluster_id": "cluster-1",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint-level: enrollment mode (reuses _resolve_verification_inputs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saliency_endpoint_enrollment_mode_requires_three_to_five_ids(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "enrollment",
            "enrollment_recording_ids": ["rec_a", "rec_b"],
            "probe_recording_id": "rec_probe",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_enrollment_mode_accepts_uploaded_files_happy_path(client):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("enrollment_files", _audio_upload("three.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    expected = {**_EXPECTED_RESULT, "reference_type": "enrollment", "cluster_id": None, "target_recording_id": None}
    with patch("app.tasks.verification.router.compute_saliency_map", return_value=expected):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={"model": "ecapa-tdnn", "reference_type": "enrollment"},
            files=files,
        )
    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.asyncio
async def test_saliency_endpoint_enrollment_mode_accepts_demo_and_session_asset_ids(client, fake_dataset_dir):
    enrollment_ids = fake_dataset_dir[:3]
    upload = await client.post(
        "/tasks/verification/session-assets/upload",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert upload.status_code == 200
    probe_id = upload.json()["asset_id"]

    expected = {
        **_EXPECTED_RESULT,
        "reference_type": "enrollment",
        "cluster_id": None,
        "target_recording_id": probe_id,
    }
    with patch("app.tasks.verification.router.compute_saliency_map", return_value=expected) as mock_compute:
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={
                "model": "ecapa-tdnn",
                "reference_type": "enrollment",
                "enrollment_recording_ids": enrollment_ids,
                "probe_recording_id": probe_id,
            },
        )

    assert response.status_code == 200
    assert response.json() == expected
    # target_recording_id is passed through as a keyword, cluster_id as None
    assert mock_compute.call_args.kwargs["target_recording_id"] == probe_id
    assert mock_compute.call_args.kwargs["cluster_id"] is None
    assert mock_compute.call_args.kwargs["reference_type"] == "enrollment"


@pytest.mark.asyncio
async def test_saliency_endpoint_enrollment_mode_rejects_ids_and_files_together(client, fake_dataset_dir):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("enrollment_files", _audio_upload("three.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={
            "model": "ecapa-tdnn",
            "reference_type": "enrollment",
            "enrollment_recording_ids": fake_dataset_dir[:3],
            "probe_recording_id": fake_dataset_dir[3],
        },
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_response_passes_through_cluster_id_and_target_id(client):
    expected = {**_EXPECTED_RESULT, "cluster_id": "cluster-7", "target_recording_id": "rec_target"}
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", return_value=expected) as mock_compute,
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={
                "model": "ecapa-tdnn",
                "reference_type": "cluster",
                "reference_recording_ids": ["rec_a", "rec_b"],
                "target_recording_id": "rec_target",
                "cluster_id": "cluster-7",
            },
        )
    assert response.status_code == 200
    assert response.json()["cluster_id"] == "cluster-7"
    assert response.json()["target_recording_id"] == "rec_target"
    assert mock_compute.call_args.kwargs["cluster_id"] == "cluster-7"
    assert mock_compute.call_args.kwargs["target_recording_id"] == "rec_target"


# ---------------------------------------------------------------------------
# Math-level: service.compute_saliency_map / _occlude_and_score
# ---------------------------------------------------------------------------


def _quarter_energy_embedding(audio_path) -> torch.Tensor:
    import torchaudio

    waveform, _ = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    samples = waveform.squeeze(0)
    n = samples.shape[0]
    quarter = n // 4
    energies = []
    for index in range(4):
        start = index * quarter
        end = n if index == 3 else (index + 1) * quarter
        chunk = samples[start:end]
        energies.append(float(chunk.abs().mean()) if chunk.numel() > 0 else 0.0)
    vector = torch.tensor(energies, dtype=torch.float32)
    norm = torch.linalg.vector_norm(vector)
    if float(norm) == 0.0:
        return torch.tensor([1.0, 0.0, 0.0, 0.0])
    return vector / norm


class _RealQuarterEnergyAdapter:
    """Deterministic, content-dependent 4-dim embedding: L2-normalized
    mean-abs-amplitude of each quarter of the (mono) waveform. Muting a
    segment therefore measurably changes both magnitude and direction,
    unlike a fixed path->tensor lookup fake."""

    def extract_embedding(self, audio_path):
        return _quarter_energy_embedding(audio_path)


_FAKE_SPEC = service.SpeakerModelSpec(
    key="test-model",
    label="Test model",
    model_id="test/model",
    revision="test-revision",
    architecture="test",
    embedding_dimension=4,
    threshold=0.5,
    recommended=True,
)


def _cosine_np(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def test_compute_saliency_map_rejects_empty_reference_list(monkeypatch):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())

    with pytest.raises(ValueError):
        service.compute_saliency_map(
            "test-model", [], "target.wav", reference_type="cluster", segment_count=4
        )


def test_compute_saliency_map_rejects_out_of_range_segment_count(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())
    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(_wav_bytes())

    with pytest.raises(ValueError):
        service.compute_saliency_map(
            "test-model", [ref_path], "target.wav", reference_type="cluster", segment_count=2
        )


def test_compute_saliency_map_rejects_empty_target_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())
    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(_wav_bytes())
    target_path = tmp_path / "empty.wav"
    target_path.write_bytes(_empty_wav_bytes())

    with pytest.raises(ValueError):
        service.compute_saliency_map(
            "test-model", [ref_path], target_path, reference_type="cluster", segment_count=4
        )


def test_compute_saliency_map_excludes_target_from_centroid(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())

    # Two references with distinct directions, and a target with a THIRD
    # distinct direction -- if the target were wrongly folded into its own
    # centroid, the resulting baseline_similarity would differ measurably
    # from the correctly-excluded value.
    ref1_path = tmp_path / "ref1.wav"
    ref1_path.write_bytes(_quarter_amplitude_wav_bytes((1.0, 0.001, 0.001, 0.001)))
    ref2_path = tmp_path / "ref2.wav"
    ref2_path.write_bytes(_quarter_amplitude_wav_bytes((0.001, 1.0, 0.001, 0.001)))
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(_quarter_amplitude_wav_bytes((0.001, 0.001, 1.0, 0.001)))

    result = service.compute_saliency_map(
        "test-model",
        [ref1_path, ref2_path],
        target_path,
        reference_type="cluster",
        segment_count=4,
    )

    ref1_embedding = _quarter_energy_embedding(ref1_path)
    ref2_embedding = _quarter_energy_embedding(ref2_path)
    target_embedding = _quarter_energy_embedding(target_path)

    correct_centroid = F.normalize(torch.stack([ref1_embedding, ref2_embedding]).mean(dim=0), p=2, dim=0)
    wrong_centroid = F.normalize(
        torch.stack([ref1_embedding, ref2_embedding, target_embedding]).mean(dim=0), p=2, dim=0
    )

    expected_correct_similarity = _cosine_np(correct_centroid, target_embedding)
    wrong_similarity = _cosine_np(wrong_centroid, target_embedding)

    assert result["baseline_similarity"] == pytest.approx(expected_correct_similarity, abs=1e-5)
    assert result["baseline_similarity"] != pytest.approx(wrong_similarity, abs=1e-3)


def test_compute_saliency_map_uses_fixed_centroid_across_all_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())

    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.1)))
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.3)))

    seen_centroids: list[torch.Tensor] = []
    original_cosine = service._cosine

    def spy_cosine(a, b):
        seen_centroids.append(a)
        return original_cosine(a, b)

    monkeypatch.setattr(service, "_cosine", spy_cosine)

    service.compute_saliency_map(
        "test-model", [ref_path], target_path, reference_type="cluster", segment_count=4
    )

    # One call for the baseline + one per segment.
    assert len(seen_centroids) == 5
    first = seen_centroids[0]
    assert all(torch.equal(first, centroid) for centroid in seen_centroids)


def test_compute_saliency_map_never_reclusters(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())

    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.1)))
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.3)))

    with patch.object(clustering, "cluster_batch") as mock_cluster_batch:
        service.compute_saliency_map(
            "test-model", [ref_path], target_path, reference_type="cluster", segment_count=4
        )
    mock_cluster_batch.assert_not_called()


def test_compute_saliency_map_similarity_change_sign_and_ranking(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())

    # Reference: uniform amplitude across all four quarters.
    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.1)))
    # Target: quarter 3 is a loud outlier; the other three match the
    # reference's uniform amplitude.
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.9)))

    reference_embedding = _quarter_energy_embedding(ref_path)
    target_embedding = _quarter_energy_embedding(target_path)
    centroid = F.normalize(reference_embedding, p=2, dim=0)  # single reference: centroid == its own embedding
    expected_baseline = _cosine_np(centroid, target_embedding)

    result = service.compute_saliency_map(
        "test-model", [ref_path], target_path, reference_type="cluster", segment_count=4
    )
    assert result["baseline_similarity"] == pytest.approx(expected_baseline, abs=1e-5)

    expected_changes = []
    for index in range(4):
        amplitudes = [0.1, 0.1, 0.1, 0.9]
        amplitudes[index] = 0.0
        occluded_path = tmp_path / f"occluded-{index}.wav"
        occluded_path.write_bytes(_quarter_amplitude_wav_bytes(tuple(amplitudes)))
        occluded_embedding = _quarter_energy_embedding(occluded_path)
        occluded_similarity = _cosine_np(centroid, occluded_embedding)
        expected_changes.append(expected_baseline - occluded_similarity)

    actual_changes = [segment["similarity_change"] for segment in result["segments"]]
    for expected, actual in zip(expected_changes, actual_changes):
        assert actual == pytest.approx(expected, abs=1e-3)

    # Muting the loud outlier quarter (index 3) should make the target look
    # MORE like the uniform reference, i.e. increase similarity, i.e. a
    # NEGATIVE similarity_change (that quarter opposed the match).
    assert expected_changes[3] < 0
    assert result["segments"][3]["similarity_change"] < 0

    # Muting a quarter that already matched the uniform reference should
    # reduce similarity, i.e. a POSITIVE similarity_change (that quarter
    # supported the match).
    assert expected_changes[0] > 0
    assert result["segments"][0]["similarity_change"] > 0

    # Ranking by influence_strength: the outlier quarter's removal should be
    # the single largest-magnitude change of the four.
    ranked = sorted(result["segments"], key=lambda s: s["influence_strength"], reverse=True)
    assert ranked[0]["segment_index"] == 4  # 1-indexed quarter 3
    for segment in result["segments"]:
        assert segment["influence_strength"] == pytest.approx(abs(segment["similarity_change"]), abs=1e-9)


def test_compute_saliency_map_response_never_contains_raw_embeddings_or_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())

    secret_dir = tmp_path / "secret_marker_xyz123"
    secret_dir.mkdir()
    ref_path = secret_dir / "ref.wav"
    ref_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.1)))
    target_path = secret_dir / "target.wav"
    target_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.3)))

    result = service.compute_saliency_map(
        "test-model", [ref_path], target_path, reference_type="cluster", segment_count=4
    )

    serialized = json.dumps(result)
    assert "secret_marker_xyz123" not in serialized
    assert "embedding" not in serialized.lower()
    assert str(ref_path) not in serialized
    assert str(target_path) not in serialized


def test_temporal_occlusion_saliency_and_compute_saliency_map_agree_on_shared_math(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _RealQuarterEnergyAdapter())

    enrollment_paths = []
    for i in range(3):
        path = tmp_path / f"enroll-{i}.wav"
        path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.1)))
        enrollment_paths.append(path)
    probe_path = tmp_path / "probe.wav"
    probe_path.write_bytes(_quarter_amplitude_wav_bytes((0.1, 0.1, 0.1, 0.3)))

    old_result = service.temporal_occlusion_saliency("test-model", enrollment_paths, probe_path, segment_count=4)
    new_result = service.compute_saliency_map(
        "test-model", enrollment_paths, probe_path, reference_type="enrollment", segment_count=4
    )

    assert old_result["baseline_similarity"] == pytest.approx(new_result["baseline_similarity"], abs=1e-9)
    for old_segment, new_segment in zip(old_result["segments"], new_result["segments"]):
        assert old_segment["start_seconds"] == pytest.approx(new_segment["start_seconds"])
        assert old_segment["end_seconds"] == pytest.approx(new_segment["end_seconds"])
        assert old_segment["occluded_similarity"] == pytest.approx(new_segment["occluded_similarity"], abs=1e-9)
        assert old_segment["importance"] == pytest.approx(new_segment["similarity_change"], abs=1e-9)
