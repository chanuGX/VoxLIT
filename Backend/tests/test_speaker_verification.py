"""Focused tests for the Speaker Verification task vertical slice."""

import io
from unittest.mock import patch

import pytest
import torch

from app.tasks.verification import service


class _FakeAdapter:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embeddings = embeddings

    def extract_embedding(self, audio_path):
        return self.embeddings[str(audio_path)]


def test_verify_speaker_uses_normalised_enrollment_centroid(monkeypatch):
    embeddings = {
        "r1.wav": torch.tensor([1.0, 0.0]),
        "r2.wav": torch.tensor([0.8, 0.2]),
        "r3.wav": torch.tensor([0.9, 0.1]),
        "probe.wav": torch.tensor([1.0, 0.0]),
    }
    fake_spec = service.SpeakerModelSpec(
        key="test-model",
        label="Test model",
        model_id="test/model",
        revision="test-revision",
        architecture="test",
        embedding_dimension=2,
        threshold=0.5,
        recommended=True,
    )
    monkeypatch.setattr(service, "get_model_spec", lambda _: fake_spec)
    monkeypatch.setattr(service, "get_model", lambda _: _FakeAdapter(embeddings))

    result = service.verify_speaker(
        "test-model",
        ["r1.wav", "r2.wav", "r3.wav"],
        "probe.wav",
    )

    assert result["same_speaker"] is True
    assert result["similarity"] > 0.99
    assert result["enrollment_count"] == 3
    assert len(result["per_reference_scores"]) == 3
    assert pytest.approx(torch.linalg.vector_norm(torch.tensor(result["enrollment_centroid"])).item()) == 1.0


def _audio_upload(name: str):
    return (name, io.BytesIO(b"RIFF-fake-audio"), "audio/wav")


@pytest.mark.asyncio
async def test_verify_endpoint_requires_three_enrollment_files(client):
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    response = await client.post(
        "/tasks/verification/verify",
        data={"model": "ecapa-tdnn"},
        files=files,
    )
    assert response.status_code == 422
    assert "3 and 5" in response.json()["detail"]


@pytest.mark.asyncio
async def test_verify_endpoint_returns_service_result(client):
    expected = {
        "model": "ecapa-tdnn",
        "similarity": 0.8,
        "threshold": 0.35,
        "same_speaker": True,
    }
    files = [
        ("enrollment_files", _audio_upload("one.wav")),
        ("enrollment_files", _audio_upload("two.wav")),
        ("enrollment_files", _audio_upload("three.wav")),
        ("probe_file", _audio_upload("probe.wav")),
    ]
    with patch("app.tasks.verification.router.verify_speaker", return_value=expected):
        response = await client.post(
            "/tasks/verification/verify",
            data={"model": "ecapa-tdnn"},
            files=files,
        )

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.asyncio
async def test_speaker_models_endpoint_excludes_xvector_baseline(client):
    response = await client.get("/tasks/verification/models")
    assert response.status_code == 200
    keys = {item["key"] for item in response.json()["models"]}
    assert keys == {"ecapa-tdnn", "resnet34-lm"}