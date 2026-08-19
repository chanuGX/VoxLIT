"""Detection metrics for Feature 1 — score distributions, DET curve, EER.

Implements SRS DF-6 (two score distributions on a common axis), DF-7 (the
Detection Error Tradeoff curve across all thresholds) and DF-8 (equal error
rate and the threshold at which it occurs).

SCORE CONVENTION
----------------
Every function here takes scores where HIGHER MEANS MORE LIKELY SPOOF, which
is what the detectors' `spoof_probability` already is. A clip is called spoof
when `score >= threshold`. From that:

    false acceptance — a spoofed clip let through as genuine (score < t)
    false rejection  — a genuine clip flagged as spoofed  (score >= t)

Note these are named from the verification system's point of view, which is
the ASVspoof convention: "accept" means "accept as genuine". Getting them the
wrong way round mirrors the DET curve and moves the EER, so the direction is
asserted in the tests.

The scores are RANKING scores, not calibrated probabilities — a threshold
derived here applies only to the dataset it came from (SRS DF-9).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetPoint:
    threshold: float
    false_acceptance_rate: float
    false_rejection_rate: float


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    eer_percent: float
    eer_threshold: float
    det_curve: list[DetPoint]
    bonafide_count: int
    spoof_count: int


class NotEnoughLabelledData(ValueError):
    """Raised when either class is missing, which makes an EER meaningless."""


def rates_at(threshold: float, bonafide: list[float], spoof: list[float]) -> tuple[float, float]:
    """(false acceptance rate, false rejection rate) at one threshold.

    Public because Feature 1 also reports the rates at the model's CURRENT
    operating threshold, not only at the EER point.
    """
    # A spoof clip scoring BELOW the threshold is wrongly accepted as genuine.
    false_acceptance = sum(1 for score in spoof if score < threshold) / len(spoof)
    # A genuine clip scoring AT OR ABOVE the threshold is wrongly rejected.
    false_rejection = sum(1 for score in bonafide if score >= threshold) / len(bonafide)
    return false_acceptance, false_rejection


def _candidate_thresholds(bonafide: list[float], spoof: list[float]) -> list[float]:
    """Every threshold at which the decision can change, plus the two ends.

    Sweeping the observed scores themselves is what makes this exact rather
    than an approximation over an arbitrary grid.
    """
    observed = sorted(set(bonafide) | set(spoof))
    # Below the lowest score everything is called genuine; above the highest,
    # everything is called spoof. Both ends are needed for a complete curve.
    lowest, highest = observed[0], observed[-1]
    span = max(highest - lowest, 1e-9)
    return [lowest - span * 1e-6, *observed, highest + span * 1e-6]


def det_curve(bonafide: list[float], spoof: list[float]) -> list[DetPoint]:
    """The full error tradeoff, one point per distinguishable threshold."""
    if not bonafide or not spoof:
        raise NotEnoughLabelledData(
            "A DET curve needs at least one genuine and one spoofed clip; "
            f"got {len(bonafide)} genuine and {len(spoof)} spoofed."
        )
    points = []
    for threshold in _candidate_thresholds(bonafide, spoof):
        far, frr = rates_at(threshold, bonafide, spoof)
        points.append(DetPoint(threshold, far, frr))
    return points


def equal_error_rate(bonafide: list[float], spoof: list[float]) -> tuple[float, float]:
    """(EER as a fraction, threshold where it occurs).

    The EER is where false acceptance and false rejection cross. With finite
    samples they rarely land exactly equal, so this takes the threshold that
    minimises the gap and reports the mean of the two rates there — the
    standard treatment.
    """
    points = det_curve(bonafide, spoof)
    best = min(points, key=lambda p: abs(p.false_acceptance_rate - p.false_rejection_rate))
    rate = (best.false_acceptance_rate + best.false_rejection_rate) / 2.0
    return rate, best.threshold


def score_histogram(scores: list[float], bins: int = 20) -> list[dict[str, float]]:
    """Counts over [0, 1], the range detector scores already live in.

    A fixed range (rather than one fitted to the data) is what lets the two
    distributions share a common axis, which is the point of DF-6.
    """
    width = 1.0 / bins
    counts = [0] * bins
    for score in scores:
        index = min(int(score / width), bins - 1)
        counts[index] += 1
    return [
        {
            "bin_start": round(index * width, 4),
            "bin_end": round((index + 1) * width, 4),
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def compute_metrics(bonafide: list[float], spoof: list[float]) -> DetectionMetrics:
    rate, threshold = equal_error_rate(bonafide, spoof)
    return DetectionMetrics(
        eer_percent=round(rate * 100, 3),
        eer_threshold=round(threshold, 6),
        det_curve=det_curve(bonafide, spoof),
        bonafide_count=len(bonafide),
        spoof_count=len(spoof),
    )
