"""Feature 1 metrics — DET curve, EER, score distributions (SRS DF-6..DF-8).

The EER is symmetric under an inverted label mapping, so it cannot police
its own correctness (SRS DF-3 says exactly this). These tests therefore pin
the DIRECTION of the error rates, not just the resulting number.
"""

from __future__ import annotations

import pytest

from app.tasks.deepfake import metrics


# --- direction ------------------------------------------------------------


def test_a_perfect_detector_has_zero_eer():
    """Genuine clips score low, spoofed high, no overlap."""
    bonafide = [0.01, 0.02, 0.05]
    spoof = [0.95, 0.98, 0.99]

    rate, threshold = metrics.equal_error_rate(bonafide, spoof)

    assert rate == 0.0
    assert 0.05 < threshold <= 0.95


def test_an_inverted_detector_has_a_terrible_eer():
    """Same scores, labels swapped: the metric must NOT look good.

    This is the check that the error rates are not mirrored. A DET
    implementation with false acceptance and rejection the wrong way round
    would report 0% here too.
    """
    bonafide = [0.95, 0.98, 0.99]
    spoof = [0.01, 0.02, 0.05]

    rate, _ = metrics.equal_error_rate(bonafide, spoof)

    assert rate == 1.0


def test_a_useless_detector_sits_near_fifty_percent():
    """Identical distributions carry no information."""
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]

    rate, _ = metrics.equal_error_rate(list(scores), list(scores))

    assert 0.4 <= rate <= 0.6


def test_false_acceptance_means_a_spoof_slipped_through():
    """At a high threshold almost everything is called genuine.

    So spoofed clips are wrongly accepted (FAR high) and genuine clips are
    all correctly accepted (FRR zero).
    """
    bonafide = [0.1, 0.2]
    spoof = [0.3, 0.4]

    points = metrics.det_curve(bonafide, spoof)
    at_high = max(points, key=lambda p: p.threshold)

    assert at_high.false_acceptance_rate == 1.0
    assert at_high.false_rejection_rate == 0.0


def test_false_rejection_means_a_genuine_clip_was_flagged():
    """At a low threshold everything is called spoof: the mirror case."""
    bonafide = [0.1, 0.2]
    spoof = [0.3, 0.4]

    points = metrics.det_curve(bonafide, spoof)
    at_low = min(points, key=lambda p: p.threshold)

    assert at_low.false_acceptance_rate == 0.0
    assert at_low.false_rejection_rate == 1.0


# --- curve shape ----------------------------------------------------------


def test_the_curve_spans_both_extremes():
    points = metrics.det_curve([0.2, 0.4], [0.6, 0.8])

    assert any(p.false_acceptance_rate == 0.0 for p in points)
    assert any(p.false_acceptance_rate == 1.0 for p in points)
    assert any(p.false_rejection_rate == 0.0 for p in points)
    assert any(p.false_rejection_rate == 1.0 for p in points)


def test_the_two_error_rates_move_in_opposite_directions():
    """Raising the threshold can only trade one error for the other."""
    points = sorted(metrics.det_curve([0.1, 0.35, 0.5], [0.4, 0.6, 0.9]),
                    key=lambda p: p.threshold)

    fars = [p.false_acceptance_rate for p in points]
    frrs = [p.false_rejection_rate for p in points]

    assert fars == sorted(fars)            # rises with the threshold
    assert frrs == sorted(frrs, reverse=True)  # falls with the threshold


def test_every_observed_score_is_a_candidate_threshold():
    """The sweep is exact, not a fixed grid that could miss the crossing."""
    bonafide = [0.123, 0.456]
    spoof = [0.789]

    thresholds = {p.threshold for p in metrics.det_curve(bonafide, spoof)}

    assert {0.123, 0.456, 0.789} <= thresholds


def test_both_classes_are_required():
    with pytest.raises(metrics.NotEnoughLabelledData):
        metrics.det_curve([], [0.5])
    with pytest.raises(metrics.NotEnoughLabelledData):
        metrics.det_curve([0.5], [])


# --- distributions --------------------------------------------------------


def test_histogram_uses_a_fixed_zero_to_one_axis():
    """DF-6 needs a COMMON axis, so the range must not follow the data."""
    tight = metrics.score_histogram([0.51, 0.52], bins=10)

    assert len(tight) == 10
    assert tight[0]["bin_start"] == 0.0
    assert tight[-1]["bin_end"] == 1.0
    assert sum(b["count"] for b in tight) == 2
    assert tight[5]["count"] == 2  # both land in [0.5, 0.6)


def test_histogram_puts_a_perfect_one_in_the_last_bin():
    """1.0 would otherwise fall off the end of the final bin."""
    histogram = metrics.score_histogram([1.0], bins=4)

    assert histogram[-1]["count"] == 1
    assert sum(b["count"] for b in histogram) == 1


def test_histogram_bins_tile_the_axis_without_gaps():
    histogram = metrics.score_histogram([], bins=5)

    for previous, following in zip(histogram, histogram[1:]):
        assert previous["bin_end"] == following["bin_start"]


# --- assembled result -----------------------------------------------------


def test_compute_metrics_reports_counts_and_a_percentage():
    result = metrics.compute_metrics([0.1, 0.2, 0.3], [0.7, 0.8])

    assert result.bonafide_count == 3
    assert result.spoof_count == 2
    assert result.eer_percent == 0.0
    assert result.det_curve
