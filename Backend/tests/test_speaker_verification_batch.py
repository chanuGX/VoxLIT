"""Focused tests for batch speaker-embedding/pairwise analysis (Commit 3)."""

import io
import json
from unittest.mock import patch

import pytest
import torch

from app.core.settings import settings
from app.tasks.verification import dataset, service

FAKE_FILES = {
    "id10018_01.wav": "Akshay_Kumar",
    "id10018_02.wav": "Akshay_Kumar",
    "id10324_01.wav": "Freida_Pinto",
    "id10324_02.wav": "Freida_Pinto",
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
    return dataset_dir


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    return _build_fake_dataset(tmp_path)


def _assert_no_ground_truth_leak(payload) -> None:
    serialized = json.dumps(payload)
    for filename, speaker in FAKE_FILES.items():
        assert filename not in serialized
        assert speaker not in serialized
    assert "metadata.csv" not in serialized
    assert "vox_indian_demo_92" not in serialized


class _FakeAdapter:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embeddings = embeddings

    def extract_embedding(self, audio_path):
        return self.embeddings[str(audio_path)]


_FAKE_SPEC = service.SpeakerModelSpec(
    key="test-model",
    label="Test model",
    model_id="test/model",
    architecture="test",
    embedding_dimension=2,
    threshold=0.5,
    recommended=True,
)


def test_extract_batch_embeddings_preserves_order(monkeypatch):
    embeddings = {
        "a.wav": torch.tensor([1.0, 0.0]),
        "b.wav": torch.tensor([0.0, 1.0]),
    }
    monkeypatch.setattr(service, "get_model", lambda _: _FakeAdapter(embeddings))

    result = service.extract_batch_embeddings("test-model", ["a.wav", "b.wav"])

    assert torch.allclose(result[0], embeddings["a.wav"])
    assert torch.allclose(result[1], embeddings["b.wav"])


def test_pairwise_similarity_matrix_is_symmetric_with_unit_diagonal():
    embeddings = torch.tensor([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]])

    similarity = service.pairwise_similarity_matrix(embeddings)

    assert similarity.shape == (3, 3)
    for i in range(3):
        assert similarity[i, i] == pytest.approx(1.0)
    assert (similarity == similarity.T).all()
    assert (similarity >= -1.0).all() and (similarity <= 1.0).all()


def test_pairwise_decision_matrix_matches_threshold():
    similarity = service.pairwise_similarity_matrix(
        torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    )

    decisions = service.pairwise_decision_matrix(similarity, threshold=0.5)

    assert decisions.dtype == bool
    for i in range(3):
        assert decisions[i, i] == True  # noqa: E712
    assert bool(decisions[0, 1]) == (similarity[0, 1] >= 0.5)
    assert bool(decisions[0, 2]) == (similarity[0, 2] >= 0.5)


def test_batch_verification_analysis_rejects_out_of_range_counts(monkeypatch):
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)

    with pytest.raises(ValueError):
        service.batch_verification_analysis("test-model", ["only-one.wav"], ["l0"])


def test_batch_verification_analysis_returns_expected_contract(monkeypatch):
    # Real adapters always return L2-normalised embeddings (via
    # `_BaseAdapter.validate_embedding`); the fake mirrors that guarantee.
    embeddings = {
        "a.wav": torch.nn.functional.normalize(torch.tensor([1.0, 0.0]), p=2, dim=0),
        "b.wav": torch.nn.functional.normalize(torch.tensor([0.9, 0.1]), p=2, dim=0),
        "c.wav": torch.nn.functional.normalize(torch.tensor([0.0, 1.0]), p=2, dim=0),
    }
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _FakeAdapter(embeddings))

    result = service.batch_verification_analysis(
        "test-model", ["a.wav", "b.wav", "c.wav"], ["l0", "l1", "l2"]
    )

    assert result["model"] == "test-model"
    assert result["recording_count"] == 3
    assert result["embedding_dimension"] == 2
    assert result["labels"] == ["l0", "l1", "l2"]
    matrix = result["similarity_matrix"]
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(matrix[1][0])
    decisions = result["decision_matrix"]
    assert decisions[0][1] == (matrix[0][1] >= result["threshold"])
    for embedding in result["embeddings"]:
        norm = sum(value**2 for value in embedding) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-5)


def test_dataset_resolve_recording_path_matches_real_file(fake_dataset_dir):
    recording = dataset.list_recordings()[0]
    path = dataset.resolve_recording_path(recording.recording_id)
    assert path.is_file()
    assert path.parent == fake_dataset_dir


@pytest.mark.parametrize(
    "bad_id",
    ["not-a-real-id", "../../etc/passwd", "id10018_01.wav", ""],
)
def test_dataset_resolve_recording_path_rejects_unknown_ids(fake_dataset_dir, bad_id):
    with pytest.raises(dataset.RecordingNotFound):
        dataset.resolve_recording_path(bad_id)


def _audio_upload(name: str):
    return (name, io.BytesIO(b"RIFF-fake-audio"), "audio/wav")


@pytest.mark.asyncio
async def test_batch_upload_endpoint_rejects_single_file(client):
    files = [("files", _audio_upload("one.wav"))]
    response = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_upload_endpoint_rejects_over_100_files(client):
    files = [("files", _audio_upload(f"f{i}.wav")) for i in range(101)]
    response = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "ecapa-tdnn"},
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_upload_endpoint_validates_count_before_saving(client):
    with patch("app.tasks.verification.router._save_upload") as mock_save:
        files = [("files", _audio_upload("only-one.wav"))]
        response = await client.post(
            "/tasks/verification/batch/upload",
            data={"model": "ecapa-tdnn"},
            files=files,
        )
    assert response.status_code == 422
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_batch_upload_endpoint_rejects_invalid_model(client):
    files = [
        ("files", _audio_upload("one.wav")),
        ("files", _audio_upload("two.wav")),
    ]
    response = await client.post(
        "/tasks/verification/batch/upload",
        data={"model": "not-a-real-model"},
        files=files,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_batch_upload_endpoint_uses_server_generated_labels(client):
    expected = {
        "model": "ecapa-tdnn",
        "labels": ["upload-000", "upload-001"],
    }
    files = [
        ("files", _audio_upload("secret_name_alice.wav")),
        ("files", _audio_upload("secret_name_bob.wav")),
    ]
    with patch(
        "app.tasks.verification.router.batch_verification_analysis",
        return_value=expected,
    ) as mock_analysis:
        response = await client.post(
            "/tasks/verification/batch/upload",
            data={"model": "ecapa-tdnn"},
            files=files,
        )

    assert response.status_code == 200
    call_labels = mock_analysis.call_args.args[2]
    assert call_labels == ["upload-000", "upload-001"]
    serialized = json.dumps(response.json())
    assert "secret_name_alice" not in serialized
    assert "secret_name_bob" not in serialized


@pytest.mark.asyncio
async def test_batch_dataset_endpoint_rejects_out_of_range_counts(client, fake_dataset_dir):
    response = await client.post(
        "/tasks/verification/batch/dataset",
        json={"model": "ecapa-tdnn", "recording_ids": ["rec_anything"]},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_dataset_endpoint_rejects_duplicates_before_resolving(client, fake_dataset_dir):
    known = dataset.list_recordings()[0].recording_id
    with patch("app.tasks.verification.router.resolve_recording_path") as mock_resolve:
        response = await client.post(
            "/tasks/verification/batch/dataset",
            json={"model": "ecapa-tdnn", "recording_ids": [known, known]},
        )
    assert response.status_code == 422
    mock_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_batch_dataset_endpoint_rejects_unknown_recording_id(client, fake_dataset_dir):
    known = dataset.list_recordings()[0].recording_id
    response = await client.post(
        "/tasks/verification/batch/dataset",
        json={"model": "ecapa-tdnn", "recording_ids": [known, "not-a-real-id"]},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_batch_dataset_endpoint_rejects_invalid_model(client, fake_dataset_dir):
    ids = [recording.recording_id for recording in dataset.list_recordings()[:2]]
    response = await client.post(
        "/tasks/verification/batch/dataset",
        json={"model": "not-a-real-model", "recording_ids": ids},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_batch_dataset_endpoint_returns_service_result_without_leaks(client, fake_dataset_dir):
    ids = [recording.recording_id for recording in dataset.list_recordings()]
    expected = {
        "model": "ecapa-tdnn",
        "labels": ids,
        "recording_count": len(ids),
    }
    with patch(
        "app.tasks.verification.router.batch_verification_analysis",
        return_value=expected,
    ) as mock_analysis:
        response = await client.post(
            "/tasks/verification/batch/dataset",
            json={"model": "ecapa-tdnn", "recording_ids": ids},
        )

    assert response.status_code == 200
    assert response.json() == expected
    call_labels = mock_analysis.call_args.args[2]
    assert call_labels == ids
    _assert_no_ground_truth_leak(response.json())
