"""Calibrated anonymous speaker clustering for batch verification.

Thresholds below are ported from the holdout-evaluated calibration artifacts
in ``Model Research/speaker_verification_evaluation/outputs/clustering/``
(commit 9aa3e0f). This module never imports from ``Model Research`` at
runtime; the calibrated numbers and their provenance are recorded here once
and for all.

Clustering thresholds are a distinct concept from pair-verification
thresholds (`SpeakerModelSpec.threshold` in ``service.py``) and must never be
derived from one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples

CLUSTERING_THRESHOLD_VERSION = "clustering-v1-holdout-ari"


@dataclass(frozen=True, slots=True)
class ClusteringSpec:
    model_key: str
    distance_threshold: float
    linkage: str
    source_holdout_ari: float
    source_holdout_nmi: float
    provenance_note: str


_PROVENANCE_NOTE = (
    "Average-linkage cosine-distance threshold calibrated on a "
    "speaker-disjoint calibration split (9 speakers, 1779 recordings) and "
    "evaluated once on a held-out split (15 speakers, 3078 recordings) from "
    "the VoxCeleb1 Indian subset. Selected by ARI (primary), NMI "
    "(tie-break), widest stable merge-height interval (tie-break), smaller "
    "threshold (final tie-break). Independent of the pair-verification EER "
    "threshold."
)

CLUSTERING_SPECS: dict[str, ClusteringSpec] = {
    "ecapa-tdnn": ClusteringSpec(
        model_key="ecapa-tdnn",
        distance_threshold=0.6493500404172964,
        linkage="average",
        source_holdout_ari=0.9996475587562372,
        source_holdout_nmi=0.9993724340688209,
        provenance_note=_PROVENANCE_NOTE,
    ),
    "resnet34-lm": ClusteringSpec(
        model_key="resnet34-lm",
        distance_threshold=0.6074221086898255,
        linkage="average",
        source_holdout_ari=0.9991962464649188,
        source_holdout_nmi=0.9986317556346057,
        provenance_note=_PROVENANCE_NOTE,
    ),
}


def similarity_to_distance(similarity: np.ndarray) -> np.ndarray:
    """Cosine distance derived from Commit 3's symmetric similarity matrix."""

    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def validate_distance_matrix(distance: np.ndarray) -> None:
    """Raise ``ValueError`` unless the matrix is a usable distance matrix."""

    if not np.all(np.isfinite(distance)):
        raise ValueError("Distance matrix contains non-finite values.")
    if not np.allclose(distance, distance.T, atol=1e-6):
        raise ValueError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(distance), 0.0, atol=1e-6):
        raise ValueError("Distance matrix must have a zero diagonal.")
    if np.any(distance < -1e-6) or np.any(distance > 2.0 + 1e-6):
        raise ValueError("Distance matrix values must be within [0, 2].")


def cluster_embeddings(distance: np.ndarray, threshold: float) -> np.ndarray:
    """Raw sklearn labels from average-linkage precomputed-distance clustering."""

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(distance)


def remap_labels_by_first_appearance(raw_labels: np.ndarray) -> list[str]:
    """Deterministically rename sklearn labels to cluster-1, cluster-2, ... by request order."""

    mapping: dict[int, str] = {}
    remapped: list[str] = []
    for raw_label in raw_labels:
        key = int(raw_label)
        if key not in mapping:
            mapping[key] = f"cluster-{len(mapping) + 1}"
        remapped.append(mapping[key])
    return remapped


def cluster_fit_scores(distance: np.ndarray, cluster_labels: Sequence[str]) -> list[float]:
    """Per-recording silhouette-style fit score; neutral 0.0 when undefined."""

    n_samples = len(cluster_labels)
    unique_clusters = set(cluster_labels)
    if len(unique_clusters) <= 1 or len(unique_clusters) == n_samples:
        return [0.0] * n_samples

    scores = silhouette_samples(distance, cluster_labels, metric="precomputed")
    return [float(score) for score in scores]


def summarize_clusters(
    cluster_labels: Sequence[str],
    similarity: np.ndarray,
    fit_scores: Sequence[float],
    member_labels: Sequence[str],
) -> list[dict[str, object]]:
    """Per-cluster membership, mean intra-cluster similarity, and mean fit score."""

    clusters: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_labels):
        clusters.setdefault(cluster_id, []).append(index)

    summaries: list[dict[str, object]] = []
    for cluster_id, indices in clusters.items():
        if len(indices) > 1:
            pair_similarities = [
                float(similarity[i, j])
                for position, i in enumerate(indices)
                for j in indices[position + 1 :]
            ]
            mean_similarity: float | None = float(np.mean(pair_similarities))
        else:
            mean_similarity = None

        summaries.append(
            {
                "cluster_id": cluster_id,
                "member_count": len(indices),
                "member_indices": indices,
                "member_labels": [member_labels[i] for i in indices],
                "mean_intra_cluster_similarity": mean_similarity,
                "mean_fit_score": float(np.mean([fit_scores[i] for i in indices])),
            }
        )
    return summaries


def cluster_batch(
    model_key: str,
    similarity: np.ndarray,
    labels: Sequence[str],
) -> dict[str, object]:
    """Anonymous average-linkage clustering, fit scores, and summaries for a batch."""

    spec = CLUSTERING_SPECS[model_key]
    distance = similarity_to_distance(similarity)
    validate_distance_matrix(distance)

    raw_labels = cluster_embeddings(distance, spec.distance_threshold)
    cluster_labels = remap_labels_by_first_appearance(raw_labels)
    fit_scores = cluster_fit_scores(distance, cluster_labels)
    summaries = summarize_clusters(cluster_labels, similarity, fit_scores, list(labels))

    return {
        "clustering_distance_threshold": spec.distance_threshold,
        "clustering_threshold_version": CLUSTERING_THRESHOLD_VERSION,
        "cluster_labels": cluster_labels,
        "cluster_fit_scores": fit_scores,
        "cluster_count": len(summaries),
        "cluster_summaries": summaries,
    }
