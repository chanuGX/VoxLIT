"""Focused tests for the visualization-only batch embedding projection (Commit 6a).

Covers the small-batch failure modes measured directly against this
environment's sklearn/umap-learn versions: PCA/t-SNE erroring when
n_samples < n_components, and UMAP erroring on 2 samples (n_neighbors
floor) or on a ValueError-incompatible TypeError from its default spectral
initialization on small batches. See Commit 6a in the approved plan for the
empirical findings that drove this design.
"""

import math

import numpy as np
import pytest

from app.tasks.verification import service

_ECAPA_DIMENSION = service.MODEL_SPECS["ecapa-tdnn"].embedding_dimension


def _make_embeddings(n: int, dimension: int = _ECAPA_DIMENSION, seed: int = 0) -> list[list[float]]:
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(n, dimension)).tolist()


# ---------------------------------------------------------------------------
# Service-level: shape contract across every supported method/size/dimension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduction_method", ["pca", "tsne", "umap"])
@pytest.mark.parametrize("n_samples", [2, 3, 10])
@pytest.mark.parametrize("n_components", [2, 3])
def test_project_batch_embeddings_returns_exact_requested_shape(
    reduction_method, n_samples, n_components
):
    embeddings = _make_embeddings(n_samples)

    coordinates, effective_components, method_used = service.project_batch_embeddings(
        embeddings, "ecapa-tdnn", reduction_method, n_components
    )

    assert coordinates.shape == (n_samples, n_components)
    assert np.isfinite(coordinates).all()
    assert 1 <= effective_components <= n_components
    assert method_used in {"pca", "tsne", "umap"}


def test_umap_falls_back_to_pca_for_two_samples():
    embeddings = _make_embeddings(2)

    for n_components in (2, 3):
        coordinates, effective_components, method_used = service.project_batch_embeddings(
            embeddings, "ecapa-tdnn", "umap", n_components
        )
        assert method_used == "pca"
        assert coordinates.shape == (2, n_components)


def test_umap_runs_natively_from_three_samples_upward():
    embeddings = _make_embeddings(3)

    _, _, method_used = service.project_batch_embeddings(
        embeddings, "ecapa-tdnn", "umap", 2
    )

    assert method_used == "umap"


@pytest.mark.parametrize("reduction_method", ["pca", "umap"])
def test_two_samples_three_components_zero_pads_the_unachievable_column(reduction_method):
    embeddings = _make_embeddings(2)

    coordinates, effective_components, _ = service.project_batch_embeddings(
        embeddings, "ecapa-tdnn", reduction_method, 3
    )

    assert effective_components == 2
    assert (coordinates[:, 2] == 0.0).all()


def test_tsne_needs_no_padding_for_two_samples_three_components():
    embeddings = _make_embeddings(2)

    coordinates, effective_components, method_used = service.project_batch_embeddings(
        embeddings, "ecapa-tdnn", "tsne", 3
    )

    assert method_used == "tsne"
    assert effective_components == 3
    assert coordinates.shape == (2, 3)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduction_method", ["pca", "tsne", "umap"])
@pytest.mark.parametrize("n_samples", [3, 10])
def test_projection_is_deterministic(reduction_method, n_samples):
    embeddings = _make_embeddings(n_samples)

    first, _, _ = service.project_batch_embeddings(embeddings, "ecapa-tdnn", reduction_method, 2)
    second, _, _ = service.project_batch_embeddings(embeddings, "ecapa-tdnn", reduction_method, 2)

    assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# Service-level input validation
# ---------------------------------------------------------------------------


def test_rejects_ragged_rows():
    embeddings = _make_embeddings(3)
    embeddings[1] = embeddings[1][:-1]

    with pytest.raises(ValueError):
        service.project_batch_embeddings(embeddings, "ecapa-tdnn", "pca", 2)


def test_rejects_empty_rows():
    embeddings = [[], []]

    with pytest.raises(ValueError):
        service.project_batch_embeddings(embeddings, "ecapa-tdnn", "pca", 2)


def test_rejects_dimension_mismatched_for_model():
    embeddings = _make_embeddings(3, dimension=10)

    with pytest.raises(ValueError):
        service.project_batch_embeddings(embeddings, "ecapa-tdnn", "pca", 2)


def test_rejects_nan_values():
    embeddings = _make_embeddings(3)
    embeddings[0][0] = math.nan

    with pytest.raises(ValueError):
        service.project_batch_embeddings(embeddings, "ecapa-tdnn", "pca", 2)


def test_rejects_infinite_values():
    embeddings = _make_embeddings(3)
    embeddings[0][0] = math.inf

    with pytest.raises(ValueError):
        service.project_batch_embeddings(embeddings, "ecapa-tdnn", "pca", 2)


def test_rejects_unsupported_model():
    embeddings = _make_embeddings(3)

    with pytest.raises(service.UnsupportedSpeakerModel):
        service.project_batch_embeddings(embeddings, "not-a-real-model", "pca", 2)


# ---------------------------------------------------------------------------
# Route-level: /tasks/verification/batch/project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_project_endpoint_returns_index_aligned_coordinates(client):
    embeddings = _make_embeddings(3)
    labels = ["rec_a", "rec_b", "rec_c"]

    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": embeddings,
            "labels": labels,
            "reduction_method": "pca",
            "n_components": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["labels"] == labels
    assert body["n_components"] == 2
    assert body["reduction_method_used"] == "pca"
    assert len(body["coordinates"]) == 3
    assert all(len(row) == 2 for row in body["coordinates"])


@pytest.mark.asyncio
async def test_batch_project_endpoint_rejects_out_of_range_counts(client):
    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": _make_embeddings(1),
            "labels": ["only-one"],
            "reduction_method": "pca",
            "n_components": 2,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_project_endpoint_rejects_mismatched_labels_length(client):
    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": _make_embeddings(3),
            "labels": ["only-two", "labels"],
            "reduction_method": "pca",
            "n_components": 2,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_project_endpoint_rejects_ragged_embeddings(client):
    embeddings = _make_embeddings(3)
    embeddings[0] = embeddings[0][:-1]

    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": embeddings,
            "labels": ["a", "b", "c"],
            "reduction_method": "pca",
            "n_components": 2,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_project_endpoint_rejects_wrong_dimension_for_model(client):
    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": _make_embeddings(3, dimension=10),
            "labels": ["a", "b", "c"],
            "reduction_method": "pca",
            "n_components": 2,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_project_endpoint_rejects_unsupported_model(client):
    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "not-a-real-model",
            "embeddings": _make_embeddings(3),
            "labels": ["a", "b", "c"],
            "reduction_method": "pca",
            "n_components": 2,
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_batch_project_endpoint_rejects_unsupported_reduction_method(client):
    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": _make_embeddings(3),
            "labels": ["a", "b", "c"],
            "reduction_method": "not-a-real-method",
            "n_components": 2,
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_batch_project_endpoint_rejects_unsupported_n_components(client):
    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": _make_embeddings(3),
            "labels": ["a", "b", "c"],
            "reduction_method": "pca",
            "n_components": 4,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_project_endpoint_umap_fallback_is_reported(client):
    response = await client.post(
        "/tasks/verification/batch/project",
        json={
            "model": "ecapa-tdnn",
            "embeddings": _make_embeddings(2),
            "labels": ["a", "b"],
            "reduction_method": "umap",
            "n_components": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reduction_method"] == "umap"
    assert body["reduction_method_used"] == "pca"
    assert body["effective_components"] == 2
    assert all(row[2] == 0.0 for row in body["coordinates"])
