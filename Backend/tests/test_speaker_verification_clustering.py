"""Focused tests for calibrated anonymous speaker clustering (Commit 4)."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from app.tasks.verification import clustering, service


def _identity_similarity(n: int) -> np.ndarray:
    return np.eye(n)


def _block_similarity() -> np.ndarray:
    # Two well-separated three-member clusters, request order 0..5.
    return np.array(
        [
            [1.00, 0.98, 0.97, 0.01, 0.02, 0.00],
            [0.98, 1.00, 0.96, 0.00, 0.01, 0.02],
            [0.97, 0.96, 1.00, 0.02, 0.00, 0.01],
            [0.01, 0.00, 0.02, 1.00, 0.97, 0.98],
            [0.02, 0.01, 0.00, 0.97, 1.00, 0.96],
            [0.00, 0.02, 0.01, 0.98, 0.96, 1.00],
        ]
    )


def test_clustering_thresholds_match_calibration_artifacts():
    assert clustering.CLUSTERING_SPECS["ecapa-tdnn"].distance_threshold == 0.6493500404172964
    assert clustering.CLUSTERING_SPECS["resnet34-lm"].distance_threshold == 0.6074221086898255


def test_similarity_to_distance_has_zero_diagonal():
    similarity = _block_similarity()
    distance = clustering.similarity_to_distance(similarity)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)


@pytest.mark.parametrize(
    "distance,message",
    [
        (np.array([[0.0, float("nan")], [float("nan"), 0.0]]), "non-finite"),
        (np.array([[0.0, 0.5], [0.9, 0.0]]), "symmetric"),
        (np.array([[0.1, 0.5], [0.5, 0.0]]), "zero diagonal"),
        (np.array([[0.0, 3.0], [3.0, 0.0]]), r"\[0, 2\]"),
    ],
)
def test_validate_distance_matrix_rejects_invalid_input(distance, message):
    with pytest.raises(ValueError, match=message):
        clustering.validate_distance_matrix(distance)


def test_validate_distance_matrix_accepts_valid_input():
    distance = clustering.similarity_to_distance(_block_similarity())
    clustering.validate_distance_matrix(distance)  # does not raise


def test_cluster_embeddings_uses_precomputed_average_linkage():
    distance = clustering.similarity_to_distance(_block_similarity())
    with patch.object(clustering, "AgglomerativeClustering", wraps=clustering.AgglomerativeClustering) as spy:
        clustering.cluster_embeddings(distance, threshold=0.5)
    _, kwargs = spy.call_args
    assert kwargs["metric"] == "precomputed"
    assert kwargs["linkage"] == "average"
    assert kwargs["n_clusters"] is None
    assert kwargs["distance_threshold"] == 0.5


def test_cluster_embeddings_finds_two_separated_clusters():
    distance = clustering.similarity_to_distance(_block_similarity())
    raw_labels = clustering.cluster_embeddings(distance, threshold=0.5)
    assert len(set(raw_labels)) == 2
    assert raw_labels[0] == raw_labels[1] == raw_labels[2]
    assert raw_labels[3] == raw_labels[4] == raw_labels[5]
    assert raw_labels[0] != raw_labels[3]


def test_remap_labels_by_first_appearance_is_deterministic():
    raw_labels = np.array([5, 5, 2, 2, 5, 9])
    remapped = clustering.remap_labels_by_first_appearance(raw_labels)
    assert remapped == [
        "cluster-1",
        "cluster-1",
        "cluster-2",
        "cluster-2",
        "cluster-1",
        "cluster-3",
    ]


def test_cluster_batch_request_order_alignment_and_labels():
    similarity = _block_similarity()
    labels = ["r0", "r1", "r2", "r3", "r4", "r5"]

    result = clustering.cluster_batch("ecapa-tdnn", similarity, labels)

    assert len(result["cluster_labels"]) == 6
    assert len(result["cluster_fit_scores"]) == 6
    assert result["cluster_labels"][0:3] == ["cluster-1"] * 3
    assert result["cluster_labels"][3:6] == ["cluster-2"] * 3
    assert result["cluster_count"] == 2
    assert result["clustering_distance_threshold"] == clustering.CLUSTERING_SPECS["ecapa-tdnn"].distance_threshold
    assert result["clustering_threshold_version"] == clustering.CLUSTERING_THRESHOLD_VERSION


def test_cluster_batch_selects_model_specific_threshold():
    similarity = _block_similarity()
    labels = ["r0", "r1", "r2", "r3", "r4", "r5"]

    ecapa_result = clustering.cluster_batch("ecapa-tdnn", similarity, labels)
    resnet_result = clustering.cluster_batch("resnet34-lm", similarity, labels)

    assert ecapa_result["clustering_distance_threshold"] != resnet_result["clustering_distance_threshold"]
    assert ecapa_result["clustering_distance_threshold"] == 0.6493500404172964
    assert resnet_result["clustering_distance_threshold"] == 0.6074221086898255


def test_cluster_fit_scores_within_valid_range():
    distance = clustering.similarity_to_distance(_block_similarity())
    cluster_labels = ["cluster-1"] * 3 + ["cluster-2"] * 3
    scores = clustering.cluster_fit_scores(distance, cluster_labels)
    assert len(scores) == 6
    assert all(-1.0 <= score <= 1.0 for score in scores)
    # Well-separated blocks should fit their own cluster well.
    assert all(score > 0.5 for score in scores)


def test_cluster_fit_scores_neutral_when_all_one_cluster():
    similarity = np.full((4, 4), 0.99)
    np.fill_diagonal(similarity, 1.0)
    distance = clustering.similarity_to_distance(similarity)
    cluster_labels = ["cluster-1"] * 4
    assert clustering.cluster_fit_scores(distance, cluster_labels) == [0.0, 0.0, 0.0, 0.0]


def test_cluster_fit_scores_neutral_when_all_singleton():
    distance = clustering.similarity_to_distance(_identity_similarity(4))
    cluster_labels = ["cluster-1", "cluster-2", "cluster-3", "cluster-4"]
    assert clustering.cluster_fit_scores(distance, cluster_labels) == [0.0, 0.0, 0.0, 0.0]


def test_cluster_fit_scores_singleton_within_mixed_result():
    similarity = np.array(
        [
            [1.00, 0.99, 0.00],
            [0.99, 1.00, 0.01],
            [0.00, 0.01, 1.00],
        ]
    )
    distance = clustering.similarity_to_distance(similarity)
    cluster_labels = ["cluster-1", "cluster-1", "cluster-2"]
    scores = clustering.cluster_fit_scores(distance, cluster_labels)
    assert len(scores) == 3
    assert all(-1.0 <= score <= 1.0 for score in scores)


@pytest.mark.parametrize(
    "similarity",
    [
        np.array([[1.0, 0.99], [0.99, 1.0]]),  # merges into one cluster
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # stays two singletons
    ],
)
def test_cluster_batch_handles_two_recording_batches(similarity):
    result = clustering.cluster_batch("ecapa-tdnn", similarity, ["a", "b"])
    assert len(result["cluster_labels"]) == 2
    assert result["cluster_fit_scores"] == [0.0, 0.0]


def test_summarize_clusters_excludes_self_similarity_and_flags_singletons():
    similarity = _block_similarity()
    cluster_labels = ["cluster-1"] * 3 + ["cluster-2", "cluster-3", "cluster-3"]
    fit_scores = [0.5, 0.6, 0.7, 0.0, 0.4, 0.4]
    member_labels = ["r0", "r1", "r2", "r3", "r4", "r5"]

    summaries = clustering.summarize_clusters(cluster_labels, similarity, fit_scores, member_labels)
    by_id = {summary["cluster_id"]: summary for summary in summaries}

    assert by_id["cluster-1"]["member_count"] == 3
    assert by_id["cluster-1"]["member_indices"] == [0, 1, 2]
    assert by_id["cluster-1"]["member_labels"] == ["r0", "r1", "r2"]
    expected_mean = (similarity[0, 1] + similarity[0, 2] + similarity[1, 2]) / 3
    assert by_id["cluster-1"]["mean_intra_cluster_similarity"] == pytest.approx(expected_mean)
    assert by_id["cluster-1"]["mean_fit_score"] == pytest.approx((0.5 + 0.6 + 0.7) / 3)

    assert by_id["cluster-2"]["member_count"] == 1
    assert by_id["cluster-2"]["mean_intra_cluster_similarity"] is None

    assert by_id["cluster-3"]["member_count"] == 2
    assert by_id["cluster-3"]["mean_intra_cluster_similarity"] == pytest.approx(similarity[4, 5])


_CLUSTERING_FAKE_SPEC = service.SpeakerModelSpec(
    key="ecapa-tdnn",
    label="Test model",
    model_id="test/model",
    architecture="test",
    embedding_dimension=2,
    threshold=0.5,
    recommended=True,
)


class _FakeAdapter:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embeddings = embeddings

    def extract_embedding(self, audio_path):
        return self.embeddings[str(audio_path)]


def test_batch_verification_analysis_includes_clustering_fields(monkeypatch):
    embeddings = {
        "a.wav": torch.nn.functional.normalize(torch.tensor([1.0, 0.0]), p=2, dim=0),
        "b.wav": torch.nn.functional.normalize(torch.tensor([0.99, 0.01]), p=2, dim=0),
        "c.wav": torch.nn.functional.normalize(torch.tensor([0.0, 1.0]), p=2, dim=0),
    }
    monkeypatch.setattr(service, "get_model_spec", lambda _: _CLUSTERING_FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _FakeAdapter(embeddings))

    result = service.batch_verification_analysis(
        "ecapa-tdnn", ["a.wav", "b.wav", "c.wav"], ["l0", "l1", "l2"]
    )

    # Existing Commit 3 fields remain untouched.
    assert result["model"] == "ecapa-tdnn"
    assert result["recording_count"] == 3
    assert "similarity_matrix" in result
    assert "decision_matrix" in result

    # New Commit 4 fields are present and aligned with request order.
    assert result["clustering_distance_threshold"] == clustering.CLUSTERING_SPECS["ecapa-tdnn"].distance_threshold
    assert result["clustering_threshold_version"] == clustering.CLUSTERING_THRESHOLD_VERSION
    assert len(result["cluster_labels"]) == 3
    assert len(result["cluster_fit_scores"]) == 3
    assert result["cluster_count"] == len(result["cluster_summaries"])


def test_pair_verification_decisions_unaffected_by_clustering(monkeypatch):
    embeddings = {
        "a.wav": torch.nn.functional.normalize(torch.tensor([1.0, 0.0]), p=2, dim=0),
        "b.wav": torch.nn.functional.normalize(torch.tensor([0.0, 1.0]), p=2, dim=0),
    }
    monkeypatch.setattr(service, "get_model_spec", lambda _: _CLUSTERING_FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: _FakeAdapter(embeddings))

    result = service.batch_verification_analysis("ecapa-tdnn", ["a.wav", "b.wav"], ["l0", "l1"])

    assert result["threshold"] == _CLUSTERING_FAKE_SPEC.threshold
    assert result["decision_matrix"] == [[True, False], [False, True]]


@pytest.mark.asyncio
async def test_batch_upload_endpoint_response_includes_clustering_fields(client):
    expected = {
        "model": "ecapa-tdnn",
        "labels": ["upload-000", "upload-001"],
        "clustering_distance_threshold": 0.6493500404172964,
        "clustering_threshold_version": "clustering-v1-holdout-ari",
        "cluster_labels": ["cluster-1", "cluster-1"],
        "cluster_fit_scores": [0.0, 0.0],
        "cluster_count": 1,
        "cluster_summaries": [
            {
                "cluster_id": "cluster-1",
                "member_count": 2,
                "member_indices": [0, 1],
                "member_labels": ["upload-000", "upload-001"],
                "mean_intra_cluster_similarity": 0.9,
                "mean_fit_score": 0.0,
            }
        ],
    }
    files = [
        ("files", ("one.wav", b"RIFF-fake-audio", "audio/wav")),
        ("files", ("two.wav", b"RIFF-fake-audio", "audio/wav")),
    ]
    with patch(
        "app.tasks.verification.router.batch_verification_analysis",
        return_value=expected,
    ):
        response = await client.post(
            "/tasks/verification/batch/upload",
            data={"model": "ecapa-tdnn"},
            files=files,
        )

    assert response.status_code == 200
    payload = response.json()
    for key in expected:
        assert payload[key] == expected[key]


@pytest.mark.asyncio
async def test_batch_dataset_endpoint_response_includes_clustering_fields_without_leaks(
    client, tmp_path, monkeypatch
):
    from app.core.settings import settings
    from app.tasks.verification import dataset

    dataset_dir = tmp_path / "vox_indian_demo_92"
    dataset_dir.mkdir()
    files = {
        "id10018_01.wav": "Akshay_Kumar",
        "id10324_01.wav": "Freida_Pinto",
    }
    for filename in files:
        (dataset_dir / filename).write_bytes(b"RIFF-fake-audio")
    (dataset_dir / "metadata.csv").write_text(
        "file_name,speaker_name\n" + "\n".join(f"{n},{s}" for n, s in files.items())
    )
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)

    ids = [recording.recording_id for recording in dataset.list_recordings()]
    expected = {
        "model": "ecapa-tdnn",
        "labels": ids,
        "recording_count": len(ids),
        "cluster_labels": ["cluster-1", "cluster-2"],
        "cluster_fit_scores": [0.0, 0.0],
        "cluster_count": 2,
        "cluster_summaries": [
            {
                "cluster_id": "cluster-1",
                "member_count": 1,
                "member_indices": [0],
                "member_labels": [ids[0]],
                "mean_intra_cluster_similarity": None,
                "mean_fit_score": 0.0,
            },
            {
                "cluster_id": "cluster-2",
                "member_count": 1,
                "member_indices": [1],
                "member_labels": [ids[1]],
                "mean_intra_cluster_similarity": None,
                "mean_fit_score": 0.0,
            },
        ],
    }
    with patch(
        "app.tasks.verification.router.batch_verification_analysis",
        return_value=expected,
    ):
        response = await client.post(
            "/tasks/verification/batch/dataset",
            json={"model": "ecapa-tdnn", "recording_ids": ids},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cluster_summaries"] == expected["cluster_summaries"]

    serialized = json.dumps(payload)
    for filename, speaker in files.items():
        assert filename not in serialized
        assert speaker not in serialized
    assert "metadata.csv" not in serialized
