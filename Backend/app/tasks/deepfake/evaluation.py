"""Feature 1 — batch-score the labelled dataset and assemble the view.

This is the ONLY place in the task that reads the bona fide/spoof answers
(`dataset.load_ground_truth`). It returns aggregates — two distributions, a
DET curve, an EER — and deliberately never returns a per-recording label, so
the workbench's per-clip exercise ("read the score, then check the protocol")
survives. See dataset.py's module docstring.

Cost: one forward pass per clip. Per-clip scores are cached under the SAME
Redis key the `/run` endpoint uses, so an evaluation warms up single-clip
lookups and vice versa, and a re-run after a failure resumes almost free.
"""

from __future__ import annotations

from starlette.concurrency import run_in_threadpool

from app.core.redis import cache_result, get_result

from . import metrics
from .dataset import (
    DATASET_ID,
    list_recordings,
    load_ground_truth,
    resolve_recording_path,
)
from .service import (
    THRESHOLD_VERSION,
    file_sha256,
    get_model_spec,
    run_detection,
)

# Demo files are static, so scores stay valid for as long as the model and
# threshold version do.
SCORE_TTL_SECONDS = 7 * 24 * 60 * 60
HISTOGRAM_BINS = 20


def per_clip_cache_key(model_key: str) -> str:
    """Shared with the /run endpoint — same key, same payload."""
    return f"df:{model_key}:{THRESHOLD_VERSION}"


async def _score_one(model_key: str, path) -> dict:
    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_key = per_clip_cache_key(model_key)

    cached = await get_result(cache_key, audio_hash)
    if cached is not None:
        return cached

    payload = await run_in_threadpool(run_detection, model_key, path)
    await cache_result(cache_key, audio_hash, payload, ttl=SCORE_TTL_SECONDS)
    return payload


def _attack_summary(rows: list[dict]) -> list[dict]:
    """Mean score per spoofing system, plus the genuine clips for contrast.

    The subset is built balanced across A07-A19 precisely so this is
    readable; a detector that collapses on one generator shows up here and
    nowhere else in the view.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        label = "bonafide" if row["label"] == "bonafide" else row["attack"]
        grouped.setdefault(label, []).append(row["score"])

    summary = [
        {
            "attack": name,
            "count": len(scores),
            "mean_score": round(sum(scores) / len(scores), 4),
            "is_spoof": name != "bonafide",
        }
        for name, scores in grouped.items()
    ]
    # Genuine first, then attacks in their catalogue order.
    summary.sort(key=lambda row: (row["is_spoof"], row["attack"]))
    return summary


async def evaluate_dataset(model_key: str, bins: int = HISTOGRAM_BINS) -> dict:
    """Score every labelled recording and assemble Feature 1's payload."""
    spec = get_model_spec(model_key)
    truth = load_ground_truth()
    recordings = list_recordings()

    rows: list[dict] = []
    for recording in recordings:
        stem = recording.display_filename.rsplit(".", 1)[0]
        entry = truth.get(stem)
        if entry is None:
            # A clip with no protocol line cannot be scored against anything.
            continue
        attack, label = entry
        payload = await _score_one(model_key, resolve_recording_path(recording.recording_id))
        rows.append({"label": label, "attack": attack, "score": payload["spoof_probability"]})

    bonafide = [row["score"] for row in rows if row["label"] == "bonafide"]
    spoof = [row["score"] for row in rows if row["label"] != "bonafide"]

    # Raises NotEnoughLabelledData if either class is missing — the router
    # turns that into a 422 rather than reporting a meaningless EER.
    computed = metrics.compute_metrics(bonafide, spoof)

    operating_far, operating_frr = metrics.rates_at(spec.threshold, bonafide, spoof)

    return {
        "model": model_key,
        "model_label": spec.label,
        "dataset_id": DATASET_ID,
        "scored": len(rows),
        "bonafide_count": computed.bonafide_count,
        "spoof_count": computed.spoof_count,
        "eer_percent": computed.eer_percent,
        "eer_threshold": computed.eer_threshold,
        # SRS DF-9: a threshold is only meaningful with its dataset attached.
        "threshold_provenance": (
            f"Equal error rate on {DATASET_ID} "
            f"({computed.bonafide_count} genuine / {computed.spoof_count} spoofed clips). "
            "Thresholds do not transfer between datasets."
        ),
        "distributions": {
            "bonafide": metrics.score_histogram(bonafide, bins=bins),
            "spoof": metrics.score_histogram(spoof, bins=bins),
        },
        "det_curve": [
            {
                "threshold": round(point.threshold, 6),
                "false_acceptance_rate": round(point.false_acceptance_rate, 6),
                "false_rejection_rate": round(point.false_rejection_rate, 6),
            }
            for point in computed.det_curve
        ],
        # Where the shipped threshold actually sits, so the gap between the
        # current operating point and the EER point is visible (SRS DF-2).
        "operating_point": {
            "threshold": spec.threshold,
            "calibrated": spec.threshold_calibrated,
            "false_acceptance_rate": round(operating_far, 6),
            "false_rejection_rate": round(operating_frr, 6),
        },
        "per_attack": _attack_summary(rows),
    }
