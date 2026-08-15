from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from evaluate_speaker_clustering_with_avg import (
    calculate_cluster_purity,
    remap_cluster_labels,
)

_DISTANCE_TOLERANCE = 1e-6
_MAX_MISMATCH_EXAMPLES = 10


def reconcile_embedding_manifest(
    manifest_ids: Sequence[str],
    embedding_stems: Sequence[str],
) -> None:
    """Fail clearly on any missing, orphan, or duplicate embedding id.

    ``embedding_stems`` must already be restricted to ``.npy`` file stems —
    any non-embedding file sitting alongside them (for example an
    ``embedding_manifest.csv`` extraction log) must be excluded by the
    caller before this check runs.
    """

    manifest_list = list(manifest_ids)
    stems_list = list(embedding_stems)

    manifest_counts = Counter(manifest_list)
    stems_counts = Counter(stems_list)

    duplicate_manifest_ids = sorted(
        item for item, count in manifest_counts.items() if count > 1
    )
    duplicate_stems = sorted(
        item for item, count in stems_counts.items() if count > 1
    )

    manifest_set = set(manifest_counts)
    stems_set = set(stems_counts)

    missing = sorted(manifest_set - stems_set)
    orphan = sorted(stems_set - manifest_set)

    problems = []

    if duplicate_manifest_ids:
        problems.append(
            f"{len(duplicate_manifest_ids)} duplicate manifest id(s): "
            f"{duplicate_manifest_ids[:_MAX_MISMATCH_EXAMPLES]}"
        )

    if duplicate_stems:
        problems.append(
            f"{len(duplicate_stems)} duplicate embedding stem(s): "
            f"{duplicate_stems[:_MAX_MISMATCH_EXAMPLES]}"
        )

    if missing:
        problems.append(
            f"{len(missing)} manifest id(s) missing an embedding: "
            f"{missing[:_MAX_MISMATCH_EXAMPLES]}"
        )

    if orphan:
        problems.append(
            f"{len(orphan)} orphan embedding(s) not present in the manifest: "
            f"{orphan[:_MAX_MISMATCH_EXAMPLES]}"
        )

    if problems:
        raise ValueError(
            "Embedding manifest reconciliation failed: " + "; ".join(problems)
        )


def validate_distance_matrix(distance_matrix: np.ndarray) -> None:
    """Raise clearly if the matrix is not a valid cosine distance matrix."""

    matrix = np.asarray(distance_matrix, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Distance matrix must be square; got shape {matrix.shape}."
        )

    if matrix.shape[0] < 1:
        raise ValueError("Distance matrix must not be empty.")

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Distance matrix contains non-finite values (NaN or Inf)."
        )

    if not np.allclose(matrix, matrix.T, atol=_DISTANCE_TOLERANCE):
        raise ValueError("Distance matrix is not symmetric.")

    diagonal = np.diagonal(matrix)

    if not np.allclose(diagonal, 0.0, atol=_DISTANCE_TOLERANCE):
        raise ValueError(
            "Distance matrix diagonal must be zero (self-distance)."
        )

    if matrix.min() < -_DISTANCE_TOLERANCE or matrix.max() > 2.0 + _DISTANCE_TOLERANCE:
        raise ValueError(
            "Distance matrix values must lie within the valid cosine "
            f"distance range [0, 2]; found range "
            f"[{matrix.min():.6f}, {matrix.max():.6f}]."
        )


def to_condensed_distance(distance_matrix: np.ndarray) -> np.ndarray:
    """Validate, then convert a square distance matrix to condensed form."""

    validate_distance_matrix(distance_matrix)

    matrix = np.asarray(distance_matrix, dtype=np.float64).copy()

    # Force exact symmetry and a zero diagonal before squareform's strict
    # checks, since validate_distance_matrix only guarantees closeness
    # within a floating-point tolerance.
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)

    return squareform(matrix, checks=True)


def build_average_linkage(condensed_distance: np.ndarray) -> np.ndarray:
    """Build the average-linkage hierarchy once, for repeated cutting."""

    return linkage(condensed_distance, method="average")


def _merge_height_intervals(Z: np.ndarray) -> list[tuple[float, float, float]]:
    """Return (representative_threshold, interval_low, interval_high) triples.

    Every distinct clustering the hierarchy can produce corresponds to
    exactly one interval between consecutive unique merge heights (plus the
    two open-ended boundary intervals below the smallest and above the
    largest merge). The representative threshold is the interval midpoint;
    the interval width is used later as the tie-break "stability" signal.
    """

    if Z.size == 0:
        return []

    merge_heights = np.unique(Z[:, 2])

    if merge_heights.size == 1:
        spread = max(float(merge_heights[0]), 1.0)
    else:
        spread = float(merge_heights[-1] - merge_heights[0])
        spread = spread if spread > 0 else 1.0

    lower_ceiling = 0.0
    upper_ceiling = float(merge_heights[-1]) + max(spread, 1.0)

    boundaries = np.concatenate(([lower_ceiling], merge_heights, [upper_ceiling]))

    intervals = []

    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        threshold = float((lower + upper) / 2.0)
        intervals.append((threshold, float(lower), float(upper)))

    return intervals


def candidate_thresholds_from_linkage(Z: np.ndarray) -> list[float]:
    """Candidate distance thresholds derived from the hierarchy's merge heights.

    Not an assumed grid — one candidate per midpoint between consecutive
    unique merge heights, plus the two boundary candidates (all-singleton
    below the smallest merge, single-cluster above the largest).
    """

    return [threshold for threshold, _, _ in _merge_height_intervals(Z)]


def cut_at_threshold(
    Z: np.ndarray,
    threshold: float,
    n_samples: int,
) -> np.ndarray:
    """Cut the precomputed hierarchy at ``threshold`` (no tree rebuild)."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    if n_samples == 1:
        return np.array([1], dtype=np.int64)

    raw_labels = fcluster(Z, t=threshold, criterion="distance")

    return remap_cluster_labels(raw_labels)


def _silhouette_or_none(
    distance_matrix: np.ndarray,
    predicted_labels: np.ndarray,
) -> float | None:
    unique_labels = np.unique(predicted_labels)

    if unique_labels.size < 2 or unique_labels.size >= len(predicted_labels):
        return None

    return float(
        silhouette_score(
            distance_matrix,
            predicted_labels,
            metric="precomputed",
        )
    )


def _partition_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict:
    return {
        "ari": float(adjusted_rand_score(true_labels, predicted_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, predicted_labels)),
        "purity": calculate_cluster_purity(true_labels, predicted_labels),
        "cluster_count": int(len(np.unique(predicted_labels))),
    }


def _select_best_row(search_rows: list[dict]) -> dict:
    def sort_key(row: dict) -> tuple[float, float, float, float]:
        return (
            row["ari"],
            row["nmi"],
            row["interval_width"],
            -row["threshold"],
        )

    return max(search_rows, key=sort_key)


def select_calibrated_threshold(
    distance_matrix: np.ndarray,
    true_labels: Sequence,
) -> dict:
    """Select a clustering distance threshold from calibration data only.

    Selection is by maximum ARI, tie-broken by NMI, then by the widest
    stable merge-height interval (more robust to small threshold
    perturbation), then by the smaller threshold (a larger threshold merges
    more points and increases false speaker merges, so it is not the
    conservative choice). Silhouette is not computed per candidate — only
    once, for the selected threshold.
    """

    true_labels_array = np.asarray(true_labels)
    n_samples = len(true_labels_array)

    if n_samples < 2:
        raise ValueError(
            "At least 2 samples are required for clustering calibration; "
            f"got {n_samples}."
        )

    condensed = to_condensed_distance(distance_matrix)
    Z = build_average_linkage(condensed)
    intervals = _merge_height_intervals(Z)

    if not intervals:
        raise ValueError(
            "No candidate thresholds could be derived from the linkage "
            "hierarchy (need at least 2 samples)."
        )

    search_rows = []

    for threshold, lower, upper in intervals:
        predicted = cut_at_threshold(Z, threshold, n_samples)
        row = {
            "threshold": threshold,
            "interval_width": float(upper - lower),
            **_partition_metrics(true_labels_array, predicted),
        }
        search_rows.append(row)

    best_row = _select_best_row(search_rows)
    selected_threshold = best_row["threshold"]

    selected_predicted = cut_at_threshold(Z, selected_threshold, n_samples)
    silhouette = _silhouette_or_none(distance_matrix, selected_predicted)

    search_rows_for_export = [
        {
            "threshold": row["threshold"],
            "ari": row["ari"],
            "nmi": row["nmi"],
            "purity": row["purity"],
            "cluster_count": row["cluster_count"],
        }
        for row in search_rows
    ]

    return {
        "selected_distance_threshold": selected_threshold,
        "candidate_count": len(search_rows),
        "search_rows": search_rows_for_export,
        "calibration_metrics": {
            "ari": best_row["ari"],
            "nmi": best_row["nmi"],
            "purity": best_row["purity"],
            "silhouette": silhouette,
            "cluster_count": best_row["cluster_count"],
        },
    }


def evaluate_threshold_once(
    distance_matrix: np.ndarray,
    true_labels: Sequence,
    threshold: float,
) -> dict:
    """Evaluate a fixed threshold once on holdout data. No search."""

    true_labels_array = np.asarray(true_labels)
    n_samples = len(true_labels_array)

    condensed = to_condensed_distance(distance_matrix)
    Z = build_average_linkage(condensed)

    predicted = cut_at_threshold(Z, threshold, n_samples)
    silhouette = _silhouette_or_none(distance_matrix, predicted)

    return {
        **_partition_metrics(true_labels_array, predicted),
        "silhouette": silhouette,
    }
