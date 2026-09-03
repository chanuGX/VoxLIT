"""Tests for dataset-level clustering evaluation metrics (SV-FR-25, SV-FR-34).

Pure calls into `evaluation_metrics.py` -- no model loading, no HTTP.
"""

import random
from dataclasses import asdict

import numpy as np
import pytest

from app.tasks.verification import evaluation_metrics


def test_worked_fixture_matches_hand_computed_values():
    # 6 recordings, 2 true groups (A: 0,1,2 / B: 3,4,5), one deliberate merge
    # error: recording 3 (true B) is grouped into predicted cluster c0 with
    # the three true-A recordings.
    ground_truth = ["A", "A", "A", "B", "B", "B"]
    predicted = ["c0", "c0", "c0", "c0", "c1", "c1"]

    result = evaluation_metrics.evaluate_predicted_clusters(predicted, ground_truth)

    # Pairs (n=6, total = 15): TP={(0,1),(0,2),(1,2),(4,5)}=4,
    # FP={(0,3),(1,3),(2,3)}=3, FN={(3,4),(3,5)}=2, TN=remaining 6.
    assert result.total_unique_pairs == 15
    assert result.true_positive_pairs == 4
    assert result.true_negative_pairs == 6
    assert result.false_positive_pairs == 3
    assert result.false_negative_pairs == 2
    assert result.pairwise_accuracy == pytest.approx(10 / 15)
    assert result.pairwise_precision == pytest.approx(4 / 7)
    assert result.pairwise_recall == pytest.approx(4 / 6)
    assert result.pairwise_f1_score == pytest.approx(8 / 13)

    # Purity: cluster c0 majority label A (3 of 4 members), cluster c1
    # majority label B (2 of 2 members) -> (3 + 2) / 6.
    assert result.cluster_purity == pytest.approx(5 / 6)

    # ARI/NMI are sklearn's trusted, permutation-invariant computation on
    # this exact fixture -- not hand-derivable arithmetic like the pairwise
    # counts, so the reference values are computed once via sklearn directly
    # and hard-coded here.
    assert result.adjusted_rand_index == pytest.approx(0.32432432432432434)
    assert result.normalized_mutual_information == pytest.approx(0.47870397138568005)


def test_perfect_agreement_with_mismatched_label_vocabularies():
    # SV-FR-25 proof: predicted cluster ids share zero string overlap with
    # ground-truth group ids, yet the partitions are identical, so every
    # metric must report a perfect match.
    predicted = ["cluster-99", "cluster-99", "cluster-42"]
    ground_truth = ["speaker-1", "speaker-1", "speaker-2"]

    result = evaluation_metrics.evaluate_predicted_clusters(predicted, ground_truth)

    assert result.adjusted_rand_index == pytest.approx(1.0)
    assert result.normalized_mutual_information == pytest.approx(1.0)
    assert result.cluster_purity == pytest.approx(1.0)
    assert result.pairwise_precision == pytest.approx(1.0)
    assert result.pairwise_recall == pytest.approx(1.0)
    assert result.pairwise_f1_score == pytest.approx(1.0)
    assert result.pairwise_accuracy == pytest.approx(1.0)
    assert result.false_positive_pairs == 0
    assert result.false_negative_pairs == 0


def test_permutation_invariance():
    predicted = ["c0", "c0", "c0", "c0", "c1", "c1"]
    ground_truth = ["A", "A", "A", "B", "B", "B"]
    baseline = evaluation_metrics.evaluate_predicted_clusters(predicted, ground_truth)

    relabeled_predicted = ["cluster-Z" if label == "c0" else "cluster-Q" for label in predicted]
    relabeled_truth = ["group-9" if label == "A" else "group-3" for label in ground_truth]
    relabeled = evaluation_metrics.evaluate_predicted_clusters(relabeled_predicted, relabeled_truth)

    assert relabeled == baseline


def test_all_singleton_predicted_clusters_never_raises_or_nans():
    predicted = [f"cluster-{i}" for i in range(6)]
    ground_truth = ["A", "A", "A", "B", "B", "B"]

    result = evaluation_metrics.evaluate_predicted_clusters(predicted, ground_truth)

    # No two recordings ever share a predicted cluster, so no pair can be a
    # true positive or false positive.
    assert result.true_positive_pairs == 0
    assert result.false_positive_pairs == 0
    assert result.pairwise_precision == 0.0
    assert result.pairwise_recall == 0.0
    assert result.pairwise_f1_score == 0.0
    assert not np.isnan(result.pairwise_precision)
    assert not np.isnan(result.pairwise_recall)


def test_single_predicted_cluster_for_everything_never_raises_or_nans():
    predicted = ["only-cluster"] * 6
    ground_truth = ["A", "A", "A", "B", "B", "B"]

    result = evaluation_metrics.evaluate_predicted_clusters(predicted, ground_truth)

    # Every pair is predicted "same", so no pair can be predicted "different"
    # -- false negatives are impossible.
    assert result.false_negative_pairs == 0
    assert 0.0 <= result.pairwise_accuracy <= 1.0
    assert 0.0 <= result.pairwise_precision <= 1.0
    assert 0.0 <= result.pairwise_recall <= 1.0
    assert 0.0 <= result.pairwise_f1_score <= 1.0
    # Majority true group (B or A, 3 members) over 6 total.
    assert result.cluster_purity == pytest.approx(0.5)


def test_length_mismatch_raises():
    with pytest.raises(evaluation_metrics.MismatchedLabelCounts) as excinfo:
        evaluation_metrics.evaluate_predicted_clusters(["c0", "c0"], ["A", "A", "B"])

    assert "2 predicted labels" in str(excinfo.value)
    assert "3 ground-truth groups" in str(excinfo.value)


def test_internal_consistency_over_random_partitions():
    rng = random.Random(1234)

    for _ in range(20):
        n = rng.randint(8, 20)
        cluster_count = rng.randint(1, 5)
        group_count = rng.randint(1, 5)
        predicted = [f"cluster-{rng.randrange(cluster_count)}" for _ in range(n)]
        ground_truth = [f"group-{rng.randrange(group_count)}" for _ in range(n)]

        result = evaluation_metrics.evaluate_predicted_clusters(predicted, ground_truth)

        expected_total = n * (n - 1) // 2
        assert result.total_unique_pairs == expected_total
        assert (
            result.true_positive_pairs
            + result.true_negative_pairs
            + result.false_positive_pairs
            + result.false_negative_pairs
            == expected_total
        )

        if result.true_positive_pairs + result.false_positive_pairs > 0:
            assert result.pairwise_precision == pytest.approx(
                result.true_positive_pairs
                / (result.true_positive_pairs + result.false_positive_pairs)
            )
        if result.true_positive_pairs + result.false_negative_pairs > 0:
            assert result.pairwise_recall == pytest.approx(
                result.true_positive_pairs
                / (result.true_positive_pairs + result.false_negative_pairs)
            )
        if result.pairwise_precision + result.pairwise_recall > 0:
            assert result.pairwise_f1_score == pytest.approx(
                2
                * result.pairwise_precision
                * result.pairwise_recall
                / (result.pairwise_precision + result.pairwise_recall)
            )

        assert 0.0 <= result.pairwise_accuracy <= 1.0
        assert 0.0 <= result.pairwise_precision <= 1.0
        assert 0.0 <= result.pairwise_recall <= 1.0
        assert 0.0 <= result.pairwise_f1_score <= 1.0
        assert 0.0 <= result.cluster_purity <= 1.0
        assert -1.0 <= result.adjusted_rand_index <= 1.0
        assert 0.0 <= result.normalized_mutual_information <= 1.0


def test_no_ground_truth_string_leakage():
    ground_truth = ["A", "A", "A", "B", "B", "B"]
    predicted = ["c0", "c0", "c0", "c0", "c1", "c1"]

    result = evaluation_metrics.evaluate_predicted_clusters(predicted, ground_truth)

    serialized = " ".join(str(value) for value in asdict(result).values())
    for group in set(ground_truth):
        assert group not in serialized
    for label in set(predicted):
        assert label not in serialized
