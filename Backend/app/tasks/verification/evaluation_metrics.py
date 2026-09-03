"""Dataset-level clustering evaluation against anonymous ground truth.

Implements SRS SV-FR-34 (ARI, NMI, purity, pairwise precision/recall/F1, and
false-positive/false-negative pair counts) in support of SV-FR-25 (predicted
clusters must be scored against ground truth using dataset-level clustering
metrics, never by comparing a cluster number to a speaker identifier).

Ported from the validated offline reference in
``Model Research/speaker_verification_evaluation/src/evaluate_speaker_clustering.py``
(``calculate_cluster_purity``, ``calculate_pairwise_metrics``). Field names
match that reference's `clustering_metrics.json` output exactly, so in-app
numbers are directly comparable to the offline evaluation.

ANONYMITY CONVENTION
---------------------
Ground-truth group labels are never compared to predicted cluster labels by
value or position -- `predicted[i] == ground_truth[i]` never happens anywhere
in this module. Every metric is either a permutation-invariant sklearn
routine (ARI, NMI) or a same/different-partition relation computed
independently within each label space (`predicted[i] == predicted[j]` and
`ground_truth[i] == ground_truth[j]`, compared to each other only as booleans,
never as strings). This module never emits a ground-truth group's actual
string value -- only aggregate counts and rates (SV-FR-25, BR-4, MQ-10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(frozen=True, slots=True)
class ClusterEvaluationMetrics:
    adjusted_rand_index: float
    normalized_mutual_information: float
    cluster_purity: float
    total_unique_pairs: int
    true_positive_pairs: int
    true_negative_pairs: int
    false_positive_pairs: int
    false_negative_pairs: int
    pairwise_accuracy: float
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1_score: float


class MismatchedLabelCounts(ValueError):
    """Raised when predicted cluster labels and ground-truth groups disagree
    in length, which makes a per-recording comparison meaningless."""


def _purity(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Weighted purity: for each predicted cluster, its members' majority
    true label counts towards the total; divided by the recording count."""

    total_majority = 0
    for cluster_id in np.unique(predicted_labels):
        members_true_labels = true_labels[predicted_labels == cluster_id]
        _, counts = np.unique(members_true_labels, return_counts=True)
        total_majority += int(counts.max())
    return total_majority / len(true_labels)


def _pairwise_counts(
    predicted_labels: np.ndarray, true_labels: np.ndarray
) -> tuple[int, int, int, int]:
    """(true_positive, true_negative, false_positive, false_negative) pair counts.

    Vectorized equivalent of enumerating every unordered pair once: an NxN
    same-predicted-cluster matrix and an NxN same-true-group matrix are each
    masked to the upper triangle (excluding the diagonal), which selects
    every one of the n*(n-1)/2 unordered pairs exactly once, then every
    selected pair is classified in one boolean pass -- so these counts and
    the rates derived from them in `_pairwise_rates` always trace to the same
    single classification, never two independently computed routes.
    """

    n = len(predicted_labels)
    same_cluster = predicted_labels[:, None] == predicted_labels[None, :]
    same_group = true_labels[:, None] == true_labels[None, :]

    upper = np.triu_indices(n, k=1)
    same_cluster_pairs = same_cluster[upper]
    same_group_pairs = same_group[upper]

    true_positive = int(np.sum(same_cluster_pairs & same_group_pairs))
    true_negative = int(np.sum(~same_cluster_pairs & ~same_group_pairs))
    false_positive = int(np.sum(same_cluster_pairs & ~same_group_pairs))
    false_negative = int(np.sum(~same_cluster_pairs & same_group_pairs))
    return true_positive, true_negative, false_positive, false_negative


def _pairwise_rates(
    true_positive: int, true_negative: int, false_positive: int, false_negative: int
) -> tuple[float, float, float, float]:
    """(accuracy, precision, recall, f1), each 0.0 on a zero denominator."""

    total_pairs = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total_pairs if total_pairs else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1_score


def evaluate_predicted_clusters(
    predicted_labels: Sequence[str],
    ground_truth_groups: Sequence[str],
) -> ClusterEvaluationMetrics:
    """Score `predicted_labels` against `ground_truth_groups` as partitions.

    Both sequences must be index-aligned and the same length. Neither is ever
    compared to the other by value -- only via ARI/NMI (sklearn, permutation
    invariant) and same/different-partition membership (see `_pairwise_counts`).
    """

    if len(predicted_labels) != len(ground_truth_groups):
        raise MismatchedLabelCounts(
            "Predicted cluster labels and ground-truth groups must have the same length; "
            f"got {len(predicted_labels)} predicted labels and "
            f"{len(ground_truth_groups)} ground-truth groups."
        )

    predicted = np.asarray(predicted_labels)
    true_labels = np.asarray(ground_truth_groups)

    adjusted_rand_index = adjusted_rand_score(true_labels, predicted)
    normalized_mutual_information = normalized_mutual_info_score(true_labels, predicted)
    cluster_purity = _purity(predicted, true_labels)

    true_positive, true_negative, false_positive, false_negative = _pairwise_counts(
        predicted, true_labels
    )
    pairwise_accuracy, pairwise_precision, pairwise_recall, pairwise_f1_score = _pairwise_rates(
        true_positive, true_negative, false_positive, false_negative
    )

    return ClusterEvaluationMetrics(
        adjusted_rand_index=float(adjusted_rand_index),
        normalized_mutual_information=float(normalized_mutual_information),
        cluster_purity=cluster_purity,
        total_unique_pairs=true_positive + true_negative + false_positive + false_negative,
        true_positive_pairs=true_positive,
        true_negative_pairs=true_negative,
        false_positive_pairs=false_positive,
        false_negative_pairs=false_negative,
        pairwise_accuracy=pairwise_accuracy,
        pairwise_precision=pairwise_precision,
        pairwise_recall=pairwise_recall,
        pairwise_f1_score=pairwise_f1_score,
    )
