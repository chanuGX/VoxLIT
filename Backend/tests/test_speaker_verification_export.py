"""Focused tests for the cluster analysis CSV export endpoint (SV-FR-36)."""

import csv
import io

import pytest

from app.tasks.verification import clustering, export, service

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac")


def _pair_and_singleton_payload() -> dict:
    """3 recordings: a 2-member cluster (real similarities) + a singleton
    (None similarities), spanning multiple opaque-id families."""

    label_a = "rec_aaaaaaaaaaaaaaaa"
    label_b = "asset_" + "b" * 32
    label_c = "upload-002"
    return {
        "model": "ecapa-tdnn",
        "labels": [label_a, label_b, label_c],
        "cluster_labels": ["cluster-1", "cluster-1", "cluster-2"],
        "cluster_summaries": [
            {
                "cluster_id": "cluster-1",
                "member_count": 2,
                "member_indices": [0, 1],
                "member_labels": [label_a, label_b],
                "mean_intra_cluster_similarity": 0.912345678,
                "min_intra_cluster_similarity": 0.912345678,
                "representative_index": 0,
                "representative_label": label_a,
                "mean_fit_score": 0.75,
            },
            {
                "cluster_id": "cluster-2",
                "member_count": 1,
                "member_indices": [2],
                "member_labels": [label_c],
                "mean_intra_cluster_similarity": None,
                "min_intra_cluster_similarity": None,
                "representative_index": 2,
                "representative_label": label_c,
                "mean_fit_score": 0.0,
            },
        ],
        "recording_cluster_stats": [
            {
                "cluster_id": "cluster-1",
                "mean_similarity_to_cluster": 0.912345678,
                "min_similarity_to_cluster": 0.912345678,
                "nearest_index": 1,
                "nearest_label": label_b,
                "nearest_similarity": 0.912345678,
                "nearest_in_same_cluster": True,
            },
            {
                "cluster_id": "cluster-1",
                "mean_similarity_to_cluster": 0.912345678,
                "min_similarity_to_cluster": 0.912345678,
                "nearest_index": 0,
                "nearest_label": label_a,
                "nearest_similarity": 0.912345678,
                "nearest_in_same_cluster": True,
            },
            {
                "cluster_id": "cluster-2",
                "mean_similarity_to_cluster": None,
                "min_similarity_to_cluster": None,
                "nearest_index": 0,
                "nearest_label": label_a,
                "nearest_similarity": 0.4,
                "nearest_in_same_cluster": False,
            },
        ],
        "cluster_count": 2,
    }


def _single_cluster_payload(size: int) -> dict:
    """`size` recordings all in one cluster -- for boundary-size tests where
    only the length check matters, not realistic clustering structure."""

    labels = [f"rec_{i:016x}" for i in range(size)]
    return {
        "model": "ecapa-tdnn",
        "labels": labels,
        "cluster_labels": ["cluster-1"] * size,
        "cluster_summaries": [
            {
                "cluster_id": "cluster-1",
                "member_count": size,
                "member_indices": list(range(size)),
                "member_labels": labels,
                "mean_intra_cluster_similarity": 0.5,
                "min_intra_cluster_similarity": 0.5,
                "representative_index": 0,
                "representative_label": labels[0],
                "mean_fit_score": 0.5,
            }
        ],
        "recording_cluster_stats": [
            {
                "cluster_id": "cluster-1",
                "mean_similarity_to_cluster": 0.5,
                "min_similarity_to_cluster": 0.5,
                "nearest_index": (i + 1) % size,
                "nearest_label": labels[(i + 1) % size],
                "nearest_similarity": 0.5,
                "nearest_in_same_cluster": True,
            }
            for i in range(size)
        ],
        "cluster_count": 1,
    }


def _parse_sections(text: str) -> tuple[dict[str, str], list[str], list[list[str]]]:
    """Split an exported CSV into (metadata dict, header row, data rows)."""

    rows = list(csv.reader(io.StringIO(text)))
    metadata: dict[str, str] = {}
    data_start = None
    for index, row in enumerate(rows):
        if not row:
            data_start = index + 1
            break
        key = row[0].removeprefix("# ")
        metadata[key] = row[1] if len(row) > 1 else ""
    assert data_start is not None, "no blank separator row found between sections"
    header = rows[data_start]
    data_rows = rows[data_start + 1 :]
    return metadata, header, data_rows


EXPECTED_HEADER = [
    "label",
    "cluster_id",
    "cluster_size",
    "is_representative",
    "mean_similarity_to_cluster",
    "min_similarity_to_cluster",
    "nearest_label",
    "nearest_similarity",
    "nearest_in_same_cluster",
]


@pytest.mark.asyncio
async def test_export_returns_csv_with_expected_header_and_rows(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")

    _metadata, header, data_rows = _parse_sections(response.text)
    assert header == EXPECTED_HEADER
    assert len(data_rows) == 3
    assert [row[0] for row in data_rows] == payload["labels"]
    assert [row[1] for row in data_rows] == payload["cluster_labels"]


@pytest.mark.asyncio
async def test_export_singleton_cluster_renders_empty_fields(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    _metadata, _header, data_rows = _parse_sections(response.text)
    singleton_row = data_rows[2]
    assert singleton_row[0] == "upload-002"
    assert singleton_row[4] == ""  # mean_similarity_to_cluster
    assert singleton_row[5] == ""  # min_similarity_to_cluster
    assert singleton_row[6] == "rec_aaaaaaaaaaaaaaaa"  # nearest_label still populated
    assert singleton_row[7] == "0.4000"


@pytest.mark.asyncio
async def test_export_exactly_one_representative_per_cluster(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    _metadata, _header, data_rows = _parse_sections(response.text)
    by_cluster: dict[str, list[list[str]]] = {}
    for row in data_rows:
        by_cluster.setdefault(row[1], []).append(row)
    for cluster_id, rows in by_cluster.items():
        representative_flags = [row[3] for row in rows]
        assert representative_flags.count("true") == 1, cluster_id
    assert data_rows[0][3] == "true"  # rec_a is cluster-1's representative
    assert data_rows[1][3] == "false"
    assert data_rows[2][3] == "true"  # sole member of cluster-2


@pytest.mark.asyncio
async def test_export_metadata_block_contains_required_fields(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    metadata, _header, _data_rows = _parse_sections(response.text)
    spec = service.MODEL_SPECS["ecapa-tdnn"]
    clustering_spec = clustering.CLUSTERING_SPECS["ecapa-tdnn"]

    assert metadata["model_key"] == "ecapa-tdnn"
    assert metadata["model_label"] == spec.label
    assert metadata["model_id"] == spec.model_id
    assert metadata["model_revision"] == spec.revision
    assert metadata["architecture"] == spec.architecture
    assert metadata["embedding_dimension"] == str(spec.embedding_dimension)
    assert metadata["pair_verification_threshold"] == f"{spec.threshold:.4f}"
    assert metadata["pair_threshold_version"] == service.PAIR_THRESHOLD_VERSION
    assert metadata["clustering_distance_threshold"] == f"{clustering_spec.distance_threshold:.4f}"
    assert metadata["clustering_threshold_version"] == clustering.CLUSTERING_THRESHOLD_VERSION
    assert metadata["clustering_linkage"] == "average"
    assert metadata["distance_metric"] == "cosine"
    assert metadata["preprocessing_version"] == service.PREPROCESSING_VERSION
    assert metadata["recording_count"] == "3"
    assert metadata["cluster_count"] == "2"
    assert metadata["export_timestamp_utc"].endswith("Z")


@pytest.mark.asyncio
async def test_export_disclaimer_states_threshold_evidence_and_anonymity(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    metadata, _header, _data_rows = _parse_sections(response.text)
    disclaimer = metadata["disclaimer"].lower()
    assert "evidence" in disclaimer
    assert "not proof" in disclaimer
    assert "anonymous" in disclaimer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("cluster_labels", p["cluster_labels"][:-1]),
        lambda p: p.__setitem__("recording_cluster_stats", p["recording_cluster_stats"][:-1]),
        lambda p: p.__setitem__("cluster_labels", p["cluster_labels"] + ["cluster-1"]),
    ],
)
async def test_export_rejects_mismatched_lengths(client, mutate):
    payload = _pair_and_singleton_payload()
    mutate(payload)
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_label",
    [
        "../etc/passwd",
        "speaker1.wav",
        "=cmd|' /C calc'!A1",
        "Akshay_Kumar",
        "",
    ],
)
async def test_export_rejects_invalid_label_formats(client, bad_label):
    payload = _pair_and_singleton_payload()
    payload["labels"][0] = bad_label
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_export_rejects_unknown_model(client):
    payload = _pair_and_singleton_payload()
    payload["model"] = "not-a-real-model"
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 422
    assert response.status_code != 400


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 101])
async def test_export_rejects_batch_size_out_of_bounds(client, size):
    payload = _single_cluster_payload(size)
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [2, 100])
async def test_export_accepts_batch_size_bounds(client, size):
    payload = _single_cluster_payload(size)
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 200
    _metadata, _header, data_rows = _parse_sections(response.text)
    assert len(data_rows) == size


@pytest.mark.asyncio
async def test_export_csv_round_trips_via_csv_reader(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    _metadata, _header, data_rows = _parse_sections(response.text)
    for index, row in enumerate(data_rows):
        stats = payload["recording_cluster_stats"][index]
        assert row[0] == payload["labels"][index]
        assert row[1] == payload["cluster_labels"][index]
        assert row[6] == stats["nearest_label"]
        assert row[8] == ("true" if stats["nearest_in_same_cluster"] else "false")


@pytest.mark.asyncio
async def test_export_no_audio_extension_appears_in_csv_text(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    lowered = response.text.lower()
    for extension in AUDIO_EXTENSIONS:
        assert extension not in lowered


@pytest.mark.asyncio
async def test_export_rejects_duplicate_cluster_id(client):
    payload = _pair_and_singleton_payload()
    payload["cluster_summaries"][1]["cluster_id"] = "cluster-1"
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_export_rejects_cluster_count_mismatch(client):
    payload = _pair_and_singleton_payload()
    payload["cluster_count"] = 3
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_export_rejects_unresolvable_cluster_id_reference(client):
    payload = _pair_and_singleton_payload()
    payload["cluster_labels"][2] = "cluster-99"
    payload["recording_cluster_stats"][2]["cluster_id"] = "cluster-99"
    response = await client.post("/tasks/verification/batch/export", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_export_content_disposition_has_no_labels_or_session_id(client):
    payload = _pair_and_singleton_payload()
    response = await client.post("/tasks/verification/batch/export", json=payload)

    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert "ecapa-tdnn" in disposition
    for label in payload["labels"]:
        assert label not in disposition
